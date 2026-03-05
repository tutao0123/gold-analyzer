"""
轻量级回测引擎：基于本地 CSV 数据滑动窗口模拟交易
输出：累计收益率、年化收益、最大回撤、夏普比率、胜率
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import pickle
from dl.trainer import GoldLSTM, compute_features, FEATURE_COLUMNS, NUM_FEATURES


class Backtester:
    """
    滑动窗口回测器（增强版）
    
    策略逻辑：
    - 每天用模型预测明天的收盘价
    - 预测涨幅 > 阈值 → 做多（满仓买入）
    - 预测跌幅 > 阈值 → 做空（反向持仓）
    - 预测幅度 < 阈值 → 观望（不操作）
    - 记录每日收益，最终统计绩效
    """
    
    def __init__(self, model_type="lstm", commodity_key="gold"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.commodity_key = commodity_key
        csv_name = "gc_f_full_history.csv" if commodity_key == "gold" else f"{commodity_key}_full_history.csv"
        self.csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", csv_name)
        self.model_type = model_type
        
    def _load_model_and_scaler(self):
        """加载模型和缩放器"""
        key = self.commodity_key
        scaler_name = "scaler.pkl" if key == "gold" else f"{key}_scaler.pkl"
        scaler_path = os.path.join(self.model_dir, scaler_name)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if self.model_type == "lstm":
            model = GoldLSTM(input_size=NUM_FEATURES).to(self.device)
            weight_file = f"{key}_lstm_weights.pth"
        elif self.model_type == "transformer":
            from dl.transformer_model import GoldTransformer
            model = GoldTransformer(input_size=NUM_FEATURES, seq_length=60).to(self.device)
            weight_file = f"{key}_transformer_weights.pth"
        else:
            raise ValueError(f"未知模型: {self.model_type}")
            
        weight_path = os.path.join(self.model_dir, weight_file)
        model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        model.eval()
        
        return model, scaler
    
    def run(self, test_days=500, seq_length=60, threshold=0.003, enable_short=True):
        """
        执行回测
        
        Args:
            test_days: 回测的交易日天数
            seq_length: 模型输入窗口长度
            threshold: 涨跌幅阈值（默认 0.3%），预测变动小于此值则观望
            enable_short: 是否启用做空机制
        """
        mode_str = "多空双向" if enable_short else "仅做多"
        print(f"=== 开始 {self.model_type.upper()} 模型回测 ({mode_str}, 阈值 {threshold:.1%}) ===")
        print(f"=== 最近 {test_days} 个交易日 ===")
        
        model, scaler = self._load_model_and_scaler()
        
        # 加载数据并计算特征
        df = pd.read_csv(self.csv_file, index_col=0, parse_dates=True)
        features_df = compute_features(df)
        
        if len(features_df) < test_days + seq_length:
            test_days = len(features_df) - seq_length - 1
            print(f"数据不足，回测窗口缩短至 {test_days} 天")
        
        all_data = features_df.values
        close_prices = features_df['Close'].values
        dates = features_df.index
        
        # 缩放
        scaled_data = scaler.transform(all_data)
        
        # 回测起点
        start_idx = len(scaled_data) - test_days
        
        daily_returns = []
        correct_directions = 0
        total_predictions = 0
        long_count = 0       # 做多次数
        short_count = 0      # 做空次数
        hold_count = 0       # 观望次数
        long_wins = 0        # 做多盈利次数
        short_wins = 0       # 做空盈利次数
        
        print(f"回测区间: {dates[start_idx].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
        
        for i in range(start_idx, len(scaled_data) - 1):
            # 取过去 seq_length 天的数据
            x_input = scaled_data[i - seq_length:i, :]
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                scaled_pred = model(x_tensor).cpu().numpy()[0][0]
            
            # 反归一化预测价格
            dummy = np.zeros((1, NUM_FEATURES))
            dummy[0, 0] = scaled_pred
            pred_price = scaler.inverse_transform(dummy)[0][0]
            
            current_price = close_prices[i]
            next_price = close_prices[i + 1]
            actual_return = (next_price - current_price) / current_price
            
            # 计算预测涨跌幅
            pred_change = (pred_price - current_price) / current_price
            
            # 策略决定
            if pred_change > threshold:
                # 预测涨幅超过阈值 → 做多
                strategy_return = actual_return
                long_count += 1
                if actual_return > 0:
                    long_wins += 1
                    
            elif pred_change < -threshold and enable_short:
                # 预测跌幅超过阈值 → 做空（收益 = -实际涨跌）
                strategy_return = -actual_return
                short_count += 1
                if actual_return < 0:
                    short_wins += 1
                    
            else:
                # 预测幅度在阈值内 → 观望
                strategy_return = 0
                hold_count += 1
            
            daily_returns.append(strategy_return)
            
            # 方向准确率（仅统计有操作的交易）
            if pred_change > threshold or (pred_change < -threshold and enable_short):
                actual_direction = 1 if next_price > current_price else -1
                pred_direction = 1 if pred_change > 0 else -1
                if pred_direction == actual_direction:
                    correct_directions += 1
                total_predictions += 1
        
        # 计算绩效指标
        daily_returns = np.array(daily_returns)
        cumulative = np.cumprod(1 + daily_returns)
        
        total_return = cumulative[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # 最大回撤
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 夏普比率 (假设无风险利率 4%)
        risk_free_daily = 0.04 / 252
        active_returns = daily_returns[daily_returns != 0]
        if len(active_returns) > 1:
            excess_returns = active_returns - risk_free_daily
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # 胜率（仅有操作的交易）
        win_rate = correct_directions / total_predictions if total_predictions > 0 else 0
        
        # 做多/做空分别胜率
        long_win_rate = long_wins / long_count if long_count > 0 else 0
        short_win_rate = short_wins / short_count if short_count > 0 else 0
        
        # 买入持有基准
        buy_hold_return = (close_prices[-1] / close_prices[start_idx]) - 1
        
        total_trades = long_count + short_count
        
        result = {
            "model": self.model_type.upper(),
            "test_days": len(daily_returns),
            "threshold": threshold,
            "enable_short": enable_short,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "long_count": long_count,
            "short_count": short_count,
            "hold_count": hold_count,
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "buy_hold_return": buy_hold_return,
        }
        
        # 打印报告
        report = (
            f"\n{'='*55}\n"
            f"  📊 {self.model_type.upper()} 模型回测报告 "
            f"({'多空双向' if enable_short else '仅做多'} | 阈值 {threshold:.1%})\n"
            f"{'='*55}\n"
            f"  回测天数:       {result['test_days']} 个交易日\n"
            f"  累计收益:       {result['total_return']:+.2%}\n"
            f"  年化收益:       {result['annual_return']:+.2%}\n"
            f"  最大回撤:       {result['max_drawdown']:.2%}\n"
            f"  夏普比率:       {result['sharpe_ratio']:.2f}\n"
            f"  {'─'*40}\n"
            f"  总交易次数:     {result['total_trades']} 次 (观望 {result['hold_count']} 天)\n"
            f"  做多: {result['long_count']}次 (胜率 {result['long_win_rate']:.1%}) | "
            f"做空: {result['short_count']}次 (胜率 {result['short_win_rate']:.1%})\n"
            f"  综合胜率:       {result['win_rate']:.1%}\n"
            f"  {'─'*40}\n"
            f"  买入持有基准:   {result['buy_hold_return']:+.2%}\n"
            f"{'='*55}\n"
        )
        print(report)
        
        return result, report
    
    def get_summary_for_agent(self, test_days=500):
        """供 Agent 调用的精简版回测摘要"""
        try:
            result, _ = self.run(test_days=test_days, threshold=0.003, enable_short=True)
            summary = (
                f"【{result['model']} 回测绩效 (近{result['test_days']}日 | 多空双向 | 阈值{result['threshold']:.1%})】\n"
                f"累计: {result['total_return']:+.2%} | 年化: {result['annual_return']:+.2%} | "
                f"回撤: {result['max_drawdown']:.2%}\n"
                f"夏普: {result['sharpe_ratio']:.2f} | 胜率: {result['win_rate']:.1%} | "
                f"交易{result['total_trades']}次 (多{result['long_count']}/空{result['short_count']})\n"
                f"基准(持有): {result['buy_hold_return']:+.2%}"
            )
            return summary
        except Exception as e:
            return f"回测执行失败: {e}"


if __name__ == "__main__":
    for model in ["lstm", "transformer"]:
        bt = Backtester(model_type=model)
        # 无阈值无做空（旧策略基线）
        bt.run(test_days=500, threshold=0.0, enable_short=False)
        # 有阈值 + 做空（增强策略）
        bt.run(test_days=500, threshold=0.003, enable_short=True)
        # 高阈值 + 做空
        bt.run(test_days=500, threshold=0.005, enable_short=True)

