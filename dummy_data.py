"""
Dummy mutual fund data with all required metrics for portfolio optimization.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_funds():
    """
    Generate dummy mutual fund data for both INR and USD currencies.
    Returns a DataFrame with all required metrics.
    """
    funds = []
    
    # Generate dates for historical returns (252 trading days per year, 3 years)
    dates = pd.date_range(end=datetime.now(), periods=756, freq='B')
    
    # INR Funds
    inr_funds = [
        # Large Cap Equity
        {"name": "HDFC Top 100 Fund", "type": "equity", "currency": "INR", "geography": "India", 
         "market_cap": "large", "volatility": 18.5, "returns": 14.2, "category": "large_cap"},
        {"name": "ICICI Prudential Bluechip Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "large", "volatility": 17.8, "returns": 13.9, "category": "large_cap"},
        {"name": "SBI Bluechip Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "large", "volatility": 19.2, "returns": 14.5, "category": "large_cap"},
        {"name": "Axis Bluechip Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "large", "volatility": 18.0, "returns": 14.0, "category": "large_cap"},
        
        # Mid Cap Equity
        {"name": "HDFC Mid-Cap Opportunities Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "mid", "volatility": 22.5, "returns": 16.8, "category": "mid_cap"},
        {"name": "SBI Magnum Midcap Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "mid", "volatility": 23.1, "returns": 17.2, "category": "mid_cap"},
        {"name": "Kotak Emerging Equity Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "mid", "volatility": 21.8, "returns": 16.5, "category": "mid_cap"},
        {"name": "DSP Midcap Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "mid", "volatility": 22.9, "returns": 17.0, "category": "mid_cap"},
        
        # Small Cap Equity
        {"name": "Nippon India Small Cap Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "small", "volatility": 28.5, "returns": 19.5, "category": "small_cap"},
        {"name": "HDFC Small Cap Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "small", "volatility": 29.2, "returns": 20.1, "category": "small_cap"},
        {"name": "SBI Small Cap Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "small", "volatility": 27.8, "returns": 19.0, "category": "small_cap"},
        {"name": "Axis Small Cap Fund", "type": "equity", "currency": "INR", "geography": "India",
         "market_cap": "small", "volatility": 28.9, "returns": 19.8, "category": "small_cap"},
        
        # Debt Funds
        {"name": "HDFC Corporate Bond Fund", "type": "debt", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 4.2, "returns": 7.5, "category": "debt"},
        {"name": "ICICI Prudential Corporate Bond Fund", "type": "debt", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 4.5, "returns": 7.8, "category": "debt"},
        {"name": "SBI Corporate Bond Fund", "type": "debt", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 4.0, "returns": 7.3, "category": "debt"},
        {"name": "Axis Corporate Debt Fund", "type": "debt", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 4.3, "returns": 7.6, "category": "debt"},
        
        # Balanced Funds
        {"name": "HDFC Balanced Advantage Fund", "type": "balanced", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 12.5, "returns": 11.8, "category": "balanced"},
        {"name": "ICICI Prudential Balanced Advantage Fund", "type": "balanced", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 13.0, "returns": 12.0, "category": "balanced"},
        {"name": "SBI Balanced Advantage Fund", "type": "balanced", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 12.8, "returns": 11.9, "category": "balanced"},
        {"name": "Axis Balanced Advantage Fund", "type": "balanced", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 12.6, "returns": 11.7, "category": "balanced"},
        
        # Tax Saver (ELSS) Funds
        {"name": "HDFC TaxSaver Fund", "type": "taxsaver", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 20.5, "returns": 15.5, "category": "tax_saver"},
        {"name": "ICICI Prudential Tax Plan", "type": "taxsaver", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 21.0, "returns": 15.8, "category": "tax_saver"},
        {"name": "SBI Long Term Equity Fund", "type": "taxsaver", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 20.8, "returns": 15.6, "category": "tax_saver"},
        {"name": "Axis Long Term Equity Fund", "type": "taxsaver", "currency": "INR", "geography": "India",
         "market_cap": None, "volatility": 20.3, "returns": 15.4, "category": "tax_saver"},
    ]
    
    # USD Funds - USA
    usd_usa_funds = [
        # Large Cap Equity
        {"name": "Vanguard S&P 500 Index Fund", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "large", "volatility": 16.5, "returns": 12.5, "category": "large_cap"},
        {"name": "Fidelity 500 Index Fund", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "large", "volatility": 16.8, "returns": 12.7, "category": "large_cap"},
        {"name": "Schwab Total Stock Market Index", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "large", "volatility": 17.0, "returns": 12.6, "category": "large_cap"},
        
        # Mid Cap Equity
        {"name": "Vanguard Mid-Cap Index Fund", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "mid", "volatility": 19.5, "returns": 14.2, "category": "mid_cap"},
        {"name": "Fidelity Mid Cap Index Fund", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "mid", "volatility": 19.8, "returns": 14.4, "category": "mid_cap"},
        
        # Small Cap Equity
        {"name": "Vanguard Small-Cap Index Fund", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "small", "volatility": 24.5, "returns": 16.8, "category": "small_cap"},
        {"name": "Fidelity Small Cap Index Fund", "type": "equity", "currency": "USD", "geography": "USA",
         "market_cap": "small", "volatility": 24.8, "returns": 17.0, "category": "small_cap"},
        
        # Debt Funds
        {"name": "Vanguard Total Bond Market Index", "type": "debt", "currency": "USD", "geography": "USA",
         "market_cap": None, "volatility": 3.5, "returns": 4.2, "category": "debt"},
        {"name": "Fidelity U.S. Bond Index Fund", "type": "debt", "currency": "USD", "geography": "USA",
         "market_cap": None, "volatility": 3.6, "returns": 4.3, "category": "debt"},
        
        # Balanced Funds
        {"name": "Vanguard Balanced Index Fund", "type": "balanced", "currency": "USD", "geography": "USA",
         "market_cap": None, "volatility": 10.5, "returns": 9.5, "category": "balanced"},
        {"name": "Fidelity Balanced Fund", "type": "balanced", "currency": "USD", "geography": "USA",
         "market_cap": None, "volatility": 10.8, "returns": 9.7, "category": "balanced"},
    ]
    
    # USD Funds - Japan
    usd_japan_funds = [
        # Large Cap Equity
        {"name": "Nikko AM Japan Equity Fund", "type": "equity", "currency": "USD", "geography": "Japan",
         "market_cap": "large", "volatility": 18.5, "returns": 8.5, "category": "large_cap"},
        {"name": "Nomura Japan Equity Fund", "type": "equity", "currency": "USD", "geography": "Japan",
         "market_cap": "large", "volatility": 18.8, "returns": 8.7, "category": "large_cap"},
        
        # Mid Cap Equity
        {"name": "Daiwa Japan Mid-Cap Fund", "type": "equity", "currency": "USD", "geography": "Japan",
         "market_cap": "mid", "volatility": 21.5, "returns": 10.2, "category": "mid_cap"},
        
        # Debt Funds
        {"name": "Nomura Japan Bond Fund", "type": "debt", "currency": "USD", "geography": "Japan",
         "market_cap": None, "volatility": 2.8, "returns": 1.5, "category": "debt"},
        
        # Balanced Funds
        {"name": "Mitsubishi Balanced Fund", "type": "balanced", "currency": "USD", "geography": "Japan",
         "market_cap": None, "volatility": 11.5, "returns": 6.5, "category": "balanced"},
    ]
    
    # USD Funds - India (for USD investors)
    usd_india_funds = [
        # Large Cap Equity
        {"name": "Franklin India Bluechip Fund (USD)", "type": "equity", "currency": "USD", "geography": "India",
         "market_cap": "large", "volatility": 19.5, "returns": 13.5, "category": "large_cap"},
        {"name": "Templeton India Growth Fund (USD)", "type": "equity", "currency": "USD", "geography": "India",
         "market_cap": "large", "volatility": 19.8, "returns": 13.7, "category": "large_cap"},
        
        # Mid Cap Equity
        {"name": "Franklin India Mid-Cap Fund (USD)", "type": "equity", "currency": "USD", "geography": "India",
         "market_cap": "mid", "volatility": 23.5, "returns": 16.2, "category": "mid_cap"},
        
        # Debt Funds
        {"name": "Franklin India Corporate Bond Fund (USD)", "type": "debt", "currency": "USD", "geography": "India",
         "market_cap": None, "volatility": 4.5, "returns": 7.2, "category": "debt"},
        
        # Balanced Funds
        {"name": "Templeton India Balanced Fund (USD)", "type": "balanced", "currency": "USD", "geography": "India",
         "market_cap": None, "volatility": 13.5, "returns": 11.5, "category": "balanced"},
    ]
    
    all_funds = inr_funds + usd_usa_funds + usd_japan_funds + usd_india_funds
    
    # Generate historical returns data for each fund
    for fund in all_funds:
        # Generate correlated returns based on volatility and expected returns
        np.random.seed(hash(fund["name"]) % 2**32)
        annual_return = fund["returns"] / 100
        annual_vol = fund["volatility"] / 100
        
        # Generate daily returns (assuming 252 trading days)
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        # Generate returns with some autocorrelation
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        returns = pd.Series(returns, index=dates)
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        fund["returns_series"] = returns
        fund["max_drawdown"] = abs(max_drawdown)
        fund["sharpe_ratio"] = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0
        
        funds.append(fund)
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "name": f["name"],
            "type": f["type"],
            "currency": f["currency"],
            "geography": f["geography"],
            "market_cap": f["market_cap"],
            "volatility": f["volatility"],
            "returns": f["returns"],
            "category": f["category"],
            "max_drawdown": f["max_drawdown"],
            "sharpe_ratio": f["sharpe_ratio"],
            "returns_series": f["returns_series"]
        }
        for f in funds
    ])
    
    return df

def get_funds_by_criteria(df, currency, category=None, geography=None):
    """Filter funds by currency, category, and geography."""
    filtered = df[df["currency"] == currency]
    
    if category:
        filtered = filtered[filtered["category"] == category]
    
    if geography:
        filtered = filtered[filtered["geography"] == geography]
    
    return filtered

