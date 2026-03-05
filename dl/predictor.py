import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

# 从 trainer 复用特征工程和模型定义
from dl.trainer import GoldLSTM, compute_features, FEATURE_COLUMNS, NUM_FEATURES


class DLPredictor:
    def __init__(self, model_dir=None, commodity_key="gold"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.commodity_key = commodity_key

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

        self.model_dir = model_dir

        # 黄金保持原有文件名，其他品种使用 {key}_full_history.csv
        if commodity_key == "gold":
            csv_name = "gc_f_full_history.csv"
        else:
            csv_name = f"{commodity_key}_full_history.csv"
        self.csv_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", csv_name
        )

        # 加载 Scaler（每个品种独立 scaler）
        if commodity_key == "gold":
            scaler_name = "scaler.pkl"
        else:
            scaler_name = f"{commodity_key}_scaler.pkl"
        scaler_path = os.path.join(model_dir, scaler_name)
        if not os.path.exists(scaler_path):
            self.is_ready = False
            return

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # 加载 LSTM
        self.lstm_model = self._load_model("lstm")

        # 尝试加载 Transformer（可选）
        self.transformer_model = self._load_model("transformer")

        self.is_ready = self.lstm_model is not None

    def _load_model(self, model_type):
        """加载指定类型的模型权重"""
        key = self.commodity_key
        if model_type == "lstm":
            weight_file = f"{key}_lstm_weights.pth"
            model = GoldLSTM(input_size=NUM_FEATURES).to(self.device)
        elif model_type == "transformer":
            weight_file = f"{key}_transformer_weights.pth"
            try:
                from dl.transformer_model import GoldTransformer
                model = GoldTransformer(input_size=NUM_FEATURES, seq_length=60).to(self.device)
            except ImportError:
                return None
        else:
            return None

        weight_path = os.path.join(self.model_dir, weight_file)
        if not os.path.exists(weight_path):
            return None

        model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def _predict_with_model(self, model, x_tensor):
        """用指定模型进行前向预测"""
        with torch.no_grad():
            scaled_pred = model(x_tensor).cpu().numpy()

        # 反归一化：构造一个与 scaler 维度匹配的假行，只填入第 0 列（Close）
        dummy = np.zeros((1, NUM_FEATURES))
        dummy[0, 0] = scaled_pred[0][0]
        actual_price = self.scaler.inverse_transform(dummy)[0][0]
        return actual_price

    def predict_next_day(self, sequence_length=60):
        if not self.is_ready:
            return (
                f"【预警】未找到 {self.commodity_key} 深度学习模型权重文件。"
                f"请先运行 python dl/trainer.py --commodity {self.commodity_key} 进行模型训练。"
            )

        try:
            if not os.path.exists(self.csv_file):
                return (
                    f"未找到本地 CSV 数据 ({self.csv_file})。"
                    f"请先运行 python dl/download_history.py --commodity {self.commodity_key}"
                )

            df = pd.read_csv(self.csv_file, index_col=0, parse_dates=True)

            # 计算多维特征
            features_df = compute_features(df)

            if len(features_df) < sequence_length:
                return "数据不足，无法进行深度学习滑动窗口预测。"

            recent_data = features_df.tail(sequence_length).values  # (60, 7)
            current_price = features_df['Close'].iloc[-1]

            # 使用训练时的同款缩放器缩放特征
            scaled_input = self.scaler.transform(recent_data)

            # 整理为 PyTorch Tensor: [1, seq_length, num_features]
            x_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)

            # LSTM 预测
            lstm_price = self._predict_with_model(self.lstm_model, x_tensor)
            lstm_diff = lstm_price - current_price
            lstm_dir = "上涨 📈" if lstm_diff > 0 else "下跌 📉"

            result = (
                f"=== 多维特征深度学习预测 ({self.commodity_key}) ===\n"
                f"输入特征: {FEATURE_COLUMNS}\n"
                f"分析窗口: 过去 {sequence_length} 个交易日\n"
                f"当前最新价: {current_price:.2f}\n\n"
                f"【LSTM 预测】\n"
                f"  目标位: {lstm_price:.2f}\n"
                f"  方向: {lstm_dir} (差值 {lstm_diff:+.2f})\n"
            )

            # Transformer 预测（如果可用）
            if self.transformer_model is not None:
                tf_price = self._predict_with_model(self.transformer_model, x_tensor)
                tf_diff = tf_price - current_price
                tf_dir = "上涨 📈" if tf_diff > 0 else "下跌 📉"

                result += (
                    f"\n【Transformer 预测】\n"
                    f"  目标位: {tf_price:.2f}\n"
                    f"  方向: {tf_dir} (差值 {tf_diff:+.2f})\n"
                )

                # 双模型共识
                if (lstm_diff > 0) == (tf_diff > 0):
                    avg_price = (lstm_price + tf_price) / 2
                    result += f"\n🔔 双模型共识: {lstm_dir.split()[0]}，均值目标位 {avg_price:.2f}"
                else:
                    result += f"\n⚠️ 双模型分歧: LSTM 看{lstm_dir.split()[0]}，Transformer 看{tf_dir.split()[0]}，建议观望"

            return result
        except Exception as e:
            return f"算法前向预测时发生内部错误：{e}"


# 用于独立调试
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="gold")
    args = parser.parse_args()
    predictor = DLPredictor(commodity_key=args.commodity)
    print(predictor.predict_next_day())
