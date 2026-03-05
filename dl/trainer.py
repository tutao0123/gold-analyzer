import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pickle

# ensure fixed device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Feature Engineering ====================

FEATURE_COLUMNS = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume']
NUM_FEATURES = len(FEATURE_COLUMNS)

def compute_features(df):
    """
    Compute 7-dimensional technical indicator features from raw OHLCV data.
    """
    df = df.copy()

    # --- RSI (14-day) ---
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
    
    # --- Bollinger Bands (20-day) ---
    sma20 = df['Close'].rolling(window=20, min_periods=1).mean()
    std20 = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    
    # --- Volume ---
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        df['Volume'] = 0
        
    # drop NaN rows produced during initial EMA/rolling calculations
    df = df.dropna()
    
    return df[FEATURE_COLUMNS]


# ==================== Dataset ====================

class GoldDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==================== LSTM Model ====================

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


# ==================== Data Preparation ====================

def prepare_data(sequence_length=60, commodity_key="gold"):
    """
    Load data from local CSV, compute multi-dimensional technical indicators,
    and build a sliding-window dataset.
    """
    if commodity_key == "gold":
        csv_name = "gc_f_full_history.csv"
    else:
        csv_name = f"{commodity_key}_full_history.csv"
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", csv_name)
    print(f"Loading {commodity_key} historical data from local CSV: {csv_file}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Local CSV not found. Please run first: python dl/download_history.py --commodity {commodity_key}"
        )
        
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # compute features
    print(f"Computing {NUM_FEATURES}-dimensional technical indicator features: {FEATURE_COLUMNS}")
    features_df = compute_features(df)
    data = features_df.values  # shape: (N, 7)

    # apply MinMaxScaler independently to each feature column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Close price is column 0; used as the Y label
    x_data, y_data = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i-sequence_length:i, :])   # (seq_len, 7)
        y_data.append(scaled_data[i, 0])                     # predict next-day close price

    x_data, y_data = np.array(x_data), np.array(y_data)

    # 80/20 train/test split
    train_size = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:train_size], y_data[:train_size]
    x_test, y_test = x_data[train_size:], y_data[train_size:]

    print(f"Dataset built: {len(x_train)} train samples, {len(x_test)} test samples, {NUM_FEATURES} features")
    return x_train, y_train, x_test, y_test, scaler


# ==================== Training Function ====================

def train_model(model_type="lstm", epochs=30, commodity_key="gold"):
    print(f"=== Starting {model_type.upper()} multi-feature deep learning training (commodity: {commodity_key}, device: {device}) ===")

    seq_length = 60
    x_train, y_train, x_test, y_test, scaler = prepare_data(
        sequence_length=seq_length, commodity_key=commodity_key
    )

    train_dataset = GoldDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = GoldDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # initialise model
    if model_type == "lstm":
        model = GoldLSTM(input_size=NUM_FEATURES).to(device)
        weight_file = f"{commodity_key}_lstm_weights.pth"
    elif model_type == "transformer":
        from dl.transformer_model import GoldTransformer
        model = GoldTransformer(input_size=NUM_FEATURES, seq_length=seq_length).to(device)
        weight_file = f"{commodity_key}_transformer_weights.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {epochs} epochs...")
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # training pass
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
        
        # validation pass
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
    
    print(f"Training complete. Best test loss: {best_test_loss:.6f}")

    # save weights and scaler
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, weight_file))
    if commodity_key == "gold":
        scaler_name = "scaler.pkl"
    else:
        scaler_name = f"{commodity_key}_scaler.pkl"
    with open(os.path.join(model_dir, scaler_name), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model weights [{weight_file}] and scaler [{scaler_name}] saved to {model_dir}/")
    return best_test_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a futures price prediction model")
    parser.add_argument("--commodity", default="gold", help="commodity key (e.g. gold, silver, copper)")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"], help="model type")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    args = parser.parse_args()
    train_model(model_type=args.model, epochs=args.epochs, commodity_key=args.commodity)
