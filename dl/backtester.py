"""
Lightweight backtesting engine: sliding-window simulated trading on local CSV data.
Outputs: cumulative return, annualised return, max drawdown, Sharpe ratio, win rate.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import pickle
from dl.trainer import PriceLSTM, compute_features, FEATURE_COLUMNS, NUM_FEATURES


class Backtester:
    """
    Sliding-window backtester (enhanced version).

    Strategy logic:
    - Each day: predict tomorrow's close price with the model
    - Predicted gain  >  threshold → go long  (full position)
    - Predicted loss  >  threshold → go short (reverse position)
    - Predicted move  <  threshold → stand aside
    - Record daily P&L, compute final performance metrics
    """

    def __init__(self, model_type="lstm", commodity_key="gold"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.commodity_key = commodity_key
        csv_name = "gc_f_full_history.csv" if commodity_key == "gold" else f"{commodity_key}_full_history.csv"
        self.csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", csv_name)
        self.model_type = model_type

    def _load_model_and_scaler(self):
        """Load model weights and scaler for the configured commodity."""
        key = self.commodity_key
        scaler_name = "scaler.pkl" if key == "gold" else f"{key}_scaler.pkl"
        scaler_path = os.path.join(self.model_dir, scaler_name)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if self.model_type == "lstm":
            model = PriceLSTM(input_size=NUM_FEATURES).to(self.device)
            weight_file = f"{key}_lstm_weights.pth"
        elif self.model_type == "transformer":
            from dl.transformer_model import PriceTransformer
            model = PriceTransformer(input_size=NUM_FEATURES, seq_length=60).to(self.device)
            weight_file = f"{key}_transformer_weights.pth"
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        weight_path = os.path.join(self.model_dir, weight_file)
        model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        model.eval()

        return model, scaler

    def run(self, test_days=500, seq_length=60, threshold=0.003, enable_short=True):
        """
        Run backtest.

        Args:
            test_days:    number of trading days to backtest
            seq_length:   model input window length
            threshold:    move threshold (default 0.3%); predicted move below this → stand aside
            enable_short: whether to allow short positions
        """
        mode_str = "long+short" if enable_short else "long-only"
        print(f"=== Starting {self.model_type.upper()} backtest ({mode_str}, threshold {threshold:.1%}) ===")
        print(f"=== Last {test_days} trading days ===")

        model, scaler = self._load_model_and_scaler()

        # load data and compute features
        df = pd.read_csv(self.csv_file, index_col=0, parse_dates=True)
        features_df = compute_features(df)

        if len(features_df) < test_days + seq_length:
            test_days = len(features_df) - seq_length - 1
            print(f"Insufficient data; backtest window shortened to {test_days} days")

        all_data = features_df.values
        close_prices = features_df['Close'].values
        dates = features_df.index

        # scale
        scaled_data = scaler.transform(all_data)

        # backtest start index
        start_idx = len(scaled_data) - test_days

        daily_returns = []
        correct_directions = 0
        total_predictions = 0
        long_count = 0
        short_count = 0
        hold_count = 0
        long_wins = 0
        short_wins = 0

        print(f"Backtest period: {dates[start_idx].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")

        for i in range(start_idx, len(scaled_data) - 1):
            # take the past seq_length days as input
            x_input = scaled_data[i - seq_length:i, :]
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)

            # predict
            with torch.no_grad():
                scaled_pred = model(x_tensor).cpu().numpy()[0][0]

            # inverse-transform predicted price
            dummy = np.zeros((1, NUM_FEATURES))
            dummy[0, 0] = scaled_pred
            pred_price = scaler.inverse_transform(dummy)[0][0]

            current_price = close_prices[i]
            next_price = close_prices[i + 1]
            actual_return = (next_price - current_price) / current_price

            # predicted percentage change
            pred_change = (pred_price - current_price) / current_price

            # strategy decision
            if pred_change > threshold:
                # predicted gain exceeds threshold → go long
                strategy_return = actual_return
                long_count += 1
                if actual_return > 0:
                    long_wins += 1

            elif pred_change < -threshold and enable_short:
                # predicted loss exceeds threshold → go short (return = -actual)
                strategy_return = -actual_return
                short_count += 1
                if actual_return < 0:
                    short_wins += 1

            else:
                # predicted move within threshold → stand aside
                strategy_return = 0
                hold_count += 1

            daily_returns.append(strategy_return)

            # directional accuracy (only trades with an active position)
            if pred_change > threshold or (pred_change < -threshold and enable_short):
                actual_direction = 1 if next_price > current_price else -1
                pred_direction = 1 if pred_change > 0 else -1
                if pred_direction == actual_direction:
                    correct_directions += 1
                total_predictions += 1

        # compute performance metrics
        daily_returns = np.array(daily_returns)
        cumulative = np.cumprod(1 + daily_returns)

        total_return = cumulative[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1

        # max drawdown
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free_daily = 0.04 / 252
        active_returns = daily_returns[daily_returns != 0]
        if len(active_returns) > 1:
            excess_returns = active_returns - risk_free_daily
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0

        # win rate (active trades only)
        win_rate = correct_directions / total_predictions if total_predictions > 0 else 0

        # long / short win rates
        long_win_rate = long_wins / long_count if long_count > 0 else 0
        short_win_rate = short_wins / short_count if short_count > 0 else 0

        # buy-and-hold benchmark
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

        # print report
        report = (
            f"\n{'='*55}\n"
            f"  📊 {self.model_type.upper()} Backtest Report "
            f"({'long+short' if enable_short else 'long-only'} | threshold {threshold:.1%})\n"
            f"{'='*55}\n"
            f"  Trading days:     {result['test_days']}\n"
            f"  Total return:     {result['total_return']:+.2%}\n"
            f"  Annual return:    {result['annual_return']:+.2%}\n"
            f"  Max drawdown:     {result['max_drawdown']:.2%}\n"
            f"  Sharpe ratio:     {result['sharpe_ratio']:.2f}\n"
            f"  {'─'*40}\n"
            f"  Total trades:     {result['total_trades']} (idle {result['hold_count']} days)\n"
            f"  Long: {result['long_count']} (win {result['long_win_rate']:.1%}) | "
            f"Short: {result['short_count']} (win {result['short_win_rate']:.1%})\n"
            f"  Overall win rate: {result['win_rate']:.1%}\n"
            f"  {'─'*40}\n"
            f"  Buy-and-hold:     {result['buy_hold_return']:+.2%}\n"
            f"{'='*55}\n"
        )
        print(report)

        return result, report

    def get_summary_for_agent(self, test_days=500):
        """Compact backtest summary for agent consumption."""
        try:
            result, _ = self.run(test_days=test_days, threshold=0.003, enable_short=True)
            summary = (
                f"[{result['model']} Backtest | last {result['test_days']} days | long+short | threshold {result['threshold']:.1%}]\n"
                f"Total: {result['total_return']:+.2%} | Annual: {result['annual_return']:+.2%} | "
                f"Drawdown: {result['max_drawdown']:.2%}\n"
                f"Sharpe: {result['sharpe_ratio']:.2f} | Win rate: {result['win_rate']:.1%} | "
                f"Trades: {result['total_trades']} (L{result['long_count']}/S{result['short_count']})\n"
                f"Benchmark (buy-and-hold): {result['buy_hold_return']:+.2%}"
            )
            return summary
        except Exception as e:
            return f"Backtest failed: {e}"


if __name__ == "__main__":
    for model in ["lstm", "transformer"]:
        bt = Backtester(model_type=model)
        # no threshold, no short (old strategy baseline)
        bt.run(test_days=500, threshold=0.0, enable_short=False)
        # with threshold + short (enhanced strategy)
        bt.run(test_days=500, threshold=0.003, enable_short=True)
        # high threshold + short
        bt.run(test_days=500, threshold=0.005, enable_short=True)
