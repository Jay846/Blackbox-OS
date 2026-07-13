import os
import csv
import json

def generate_mock_data(base_dir="blackbox_os/roles/data_scientist/workflows/mock_data"):
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. grid_bot_fills.csv
    # 6 wins (average 160.0), 4 losses (average -90.0) -> win_rate=0.6, payoff=1.777778
    grid_fills = [
        {"timestamp": "2026-07-09T10:00:00Z", "symbol": "BTC/USDT", "side": "buy", "price": 90000.0, "amount": 0.1, "realized_pnl": 150.0},
        {"timestamp": "2026-07-09T11:00:00Z", "symbol": "BTC/USDT", "side": "sell", "price": 91000.0, "amount": 0.1, "realized_pnl": 180.0},
        {"timestamp": "2026-07-09T12:00:00Z", "symbol": "BTC/USDT", "side": "buy", "price": 89500.0, "amount": 0.1, "realized_pnl": -100.0},
        {"timestamp": "2026-07-09T13:00:00Z", "symbol": "BTC/USDT", "side": "sell", "price": 90500.0, "amount": 0.1, "realized_pnl": 120.0},
        {"timestamp": "2026-07-09T14:00:00Z", "symbol": "BTC/USDT", "side": "buy", "price": 89000.0, "amount": 0.1, "realized_pnl": -80.0},
        {"timestamp": "2026-07-09T15:00:00Z", "symbol": "BTC/USDT", "side": "sell", "price": 91500.0, "amount": 0.1, "realized_pnl": 210.0},
        {"timestamp": "2026-07-09T16:00:00Z", "symbol": "BTC/USDT", "side": "buy", "price": 88500.0, "amount": 0.1, "realized_pnl": -120.0},
        {"timestamp": "2026-07-09T17:00:00Z", "symbol": "BTC/USDT", "side": "sell", "price": 90000.0, "amount": 0.1, "realized_pnl": 160.0},
        {"timestamp": "2026-07-09T18:00:00Z", "symbol": "BTC/USDT", "side": "buy", "price": 89800.0, "amount": 0.1, "realized_pnl": -60.0},
        {"timestamp": "2026-07-09T19:00:00Z", "symbol": "BTC/USDT", "side": "sell", "price": 91200.0, "amount": 0.1, "realized_pnl": 140.0},
    ]
    with open(os.path.join(base_dir, "grid_bot_fills.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_fills[0].keys())
        writer.writeheader()
        writer.writerows(grid_fills)
        
    # 2. eth_usd_prices.csv
    # 15 rows with True Range = 10.0
    eth_prices = []
    close_val = 3000.0
    for i in range(15):
        high_val = close_val + 5.0
        low_val = close_val - 5.0
        eth_prices.append({
            "timestamp": f"2026-07-09T{10+i:02d}:00:00Z",
            "high": high_val,
            "low": low_val,
            "close": close_val
        })
        # Keep close constant to keep TR constant at 10.0
    with open(os.path.join(base_dir, "eth_usd_prices.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eth_prices[0].keys())
        writer.writeheader()
        writer.writerows(eth_prices)
        
    # 3. trade_performance.json
    trade_perf = {
        "trades": [
            {"trade_id": 1, "return": 200.0},
            {"trade_id": 2, "return": 200.0},
            {"trade_id": 3, "return": -100.0},
            {"trade_id": 4, "return": 200.0},
            {"trade_id": 5, "return": -100.0},
            {"trade_id": 6, "return": 200.0},
            {"trade_id": 7, "return": -100.0},
            {"trade_id": 8, "return": 200.0},
            {"trade_id": 9, "return": -100.0},
            {"trade_id": 10, "return": 200.0},
        ]
    }
    with open(os.path.join(base_dir, "trade_performance.json"), "w") as f:
        json.dump(trade_perf, f, indent=2)
        
    # 4. features.csv
    features = [
        {"column_name": "timestamp", "formula": "raw_time"},
        {"column_name": "price", "formula": "price_t"},
        {"column_name": "lagged_price_1", "formula": "price_t minus 1"},
        {"column_name": "future_return", "formula": "price_{t+1} / price_t - 1"},
        {"column_name": "total_revenue_offset", "formula": "target * 5 + 10"}
    ]
    with open(os.path.join(base_dir, "features.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=features[0].keys())
        writer.writeheader()
        writer.writerows(features)
        
    # 5. raw_returns.csv
    raw_ret = [
        {"return": 12.0},
        {"return": 17.0},
        {"return": 33.0},
        {"return": 44.0},
        {"return": 51.0}
    ]
    with open(os.path.join(base_dir, "raw_returns.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_ret[0].keys())
        writer.writeheader()
        writer.writerows(raw_ret)
        
    # 6. drift_metrics.json
    drift_met = {
        "metrics": {
            "implied_vol": {
                "ks_score": 0.35,
                "warning_threshold": 0.25
            }
        }
    }
    with open(os.path.join(base_dir, "drift_metrics.json"), "w") as f:
        json.dump(drift_met, f, indent=2)
        
    print(f"Mock data files generated successfully in: {base_dir}")

if __name__ == "__main__":
    generate_mock_data()
