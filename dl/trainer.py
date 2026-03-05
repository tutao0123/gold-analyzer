import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pickle

# 保证固定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 特征工程 ====================

FEATURE_COLUMNS = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume']
NUM_FEATURES = len(FEATURE_COLUMNS)

def compute_features(df):
    """
    从原始 OHLCV 数据计算 7 维技术指标特征
    """
    df = df.copy()
    
    # --- RSI (14日) ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- MACD (12, 26, 9) ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # --- 布林带 (20日) ---
    sma20 = df['Close'].rolling(window=20, min_periods=1).mean()
    std20 = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    
    # --- Volume ---
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        df['Volume'] = 0
        
    # 清除 NaN（EMA 等计算初期会产生）
    df = df.dropna()
    
    return df[FEATURE_COLUMNS]


# ==================== 数据集 ====================

class GoldDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==================== LSTM 模型 ====================

class GoldLSTM(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, hidden_size=128, num_layers=2, dropout=0.2):
        super(GoldLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ==================== 数据准备 ====================

def prepare_data(sequence_length=60, commodity_key="gold"):
    """
    从本地 CSV 加载数据，计算多维技术指标，构建滑动窗口数据集
    """
    if commodity_key == "gold":
        csv_name = "gc_f_full_history.csv"
    else:
        csv_name = f"{commodity_key}_full_history.csv"
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", csv_name)
    print(f"正在从本地 CSV 加载 {commodity_key} 历史数据: {csv_file}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"未找到本地 CSV 数据。请先运行: python dl/download_history.py --commodity {commodity_key}"
        )
        
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # 计算 7 维特征
    print(f"正在计算 {NUM_FEATURES} 维技术指标特征: {FEATURE_COLUMNS}")
    features_df = compute_features(df)
    data = features_df.values  # shape: (N, 7)
    
    # 对每个特征列分别做 MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 收盘价在第 0 列，用于构建 Y 标签
    x_data, y_data = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i-sequence_length:i, :])   # (seq_len, 7)
        y_data.append(scaled_data[i, 0])                     # 预测下一天收盘价
        
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    # 80/20 拆分
    train_size = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:train_size], y_data[:train_size]
    x_test, y_test = x_data[train_size:], y_data[train_size:]
    
    print(f"数据集构建完成: 训练集 {len(x_train)} 条, 测试集 {len(x_test)} 条, 特征维度 {NUM_FEATURES}")
    return x_train, y_train, x_test, y_test, scaler


# ==================== 训练函数 ====================

def train_model(model_type="lstm", epochs=30, commodity_key="gold"):
    print(f"=== 开始 {model_type.upper()} 多维特征深度学习训练 (品种: {commodity_key}, 设备: {device}) ===")

    seq_length = 60
    x_train, y_train, x_test, y_test, scaler = prepare_data(
        sequence_length=seq_length, commodity_key=commodity_key
    )

    train_dataset = GoldDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = GoldDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    if model_type == "lstm":
        model = GoldLSTM(input_size=NUM_FEATURES).to(device)
        weight_file = f"{commodity_key}_lstm_weights.pth"
    elif model_type == "transformer":
        from dl.transformer_model import GoldTransformer
        model = GoldTransformer(input_size=NUM_FEATURES, seq_length=seq_length).to(device)
        weight_file = f"{commodity_key}_transformer_weights.pth"
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"开始训练 {epochs} 轮...")
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch).squeeze()
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * x_batch.size(0)
        test_loss /= len(test_loader.dataset)
        
        scheduler.step(test_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
        if test_loss < best_test_loss:
            best_test_loss = test_loss
    
    print(f"训练完毕！最佳测试 Loss: {best_test_loss:.6f}")
    
    # 保存
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, weight_file))
    if commodity_key == "gold":
        scaler_name = "scaler.pkl"
    else:
        scaler_name = f"{commodity_key}_scaler.pkl"
    with open(os.path.join(model_dir, scaler_name), "wb") as f:
        pickle.dump(scaler, f)

    print(f"模型权重 [{weight_file}] 与 Scaler [{scaler_name}] 已保存至 {model_dir}/")
    return best_test_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练期货价格预测模型")
    parser.add_argument("--commodity", default="gold", help="品种 key（如 gold, silver, copper）")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"], help="模型类型")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    args = parser.parse_args()
    train_model(model_type=args.model, epochs=args.epochs, commodity_key=args.commodity)
