# risk.py
import numpy as np
import pandas as pd

def monte_carlo_risk_simulation(data, n_simulations=1000):
    """
    ใช้ Monte Carlo Simulation เพื่อประเมินความผันผวนและ maximum drawdown
    """
    returns = data['close'].pct_change().dropna().values
    simulated_drawdowns = []
    for _ in range(n_simulations):
        simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
        simulated_equity = np.cumprod(1 + simulated_returns)
        peak = np.maximum.accumulate(simulated_equity)
        drawdown = (peak - simulated_equity) / peak
        simulated_drawdowns.append(np.max(drawdown))
    avg_drawdown = np.mean(simulated_drawdowns)
    return avg_drawdown

def calculate_position_size(capital, risk_percent, entry_price, SL, volatility_factor=1.0):
    """
    คำนวณขนาดตำแหน่งโดยใช้ capital, risk_percent, entry_price, SL และปรับด้วย volatility_factor
    """
    risk_amount = capital * risk_percent
    stop_loss_distance = abs(entry_price - SL) * volatility_factor
    if stop_loss_distance == 0:
        return 0
    position_size = risk_amount / stop_loss_distance
    return position_size
