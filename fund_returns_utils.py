"""
Utility functions to calculate and display annual returns from daily returns series.
"""
import pandas as pd
import numpy as np
from dummy_data import generate_dummy_funds

def calculate_annual_returns(returns_series):
    """
    Calculate annual returns from daily returns series.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns series with datetime index
    
    Returns:
    --------
    dict : Annual returns by year
    """
    if returns_series is None or len(returns_series) == 0:
        return {}
    
    # Group by year
    annual_returns = {}
    returns_series.index = pd.to_datetime(returns_series.index)
    
    for year in returns_series.index.year.unique():
        year_returns = returns_series[returns_series.index.year == year]
        if len(year_returns) > 0:
            # Calculate cumulative return for the year
            cumulative = (1 + year_returns).prod() - 1
            annual_returns[str(year)] = round(cumulative * 100, 2)
    
    return annual_returns

def get_fund_annual_returns(fund_name=None, currency=None, category=None):
    """
    Get annual returns for funds.
    
    Parameters:
    -----------
    fund_name : str, optional
        Specific fund name
    currency : str, optional
        Filter by currency
    category : str, optional
        Filter by category
    
    Returns:
    --------
    dict : Fund annual returns data
    """
    df = generate_dummy_funds()
    
    # Filter if needed
    if currency:
        df = df[df["currency"] == currency]
    if category:
        df = df[df["category"] == category]
    if fund_name:
        df = df[df["name"] == fund_name]
    
    results = {}
    for _, row in df.iterrows():
        returns_series = row["returns_series"]
        annual_returns = calculate_annual_returns(returns_series)
        
        results[row["name"]] = {
            "name": row["name"],
            "currency": row["currency"],
            "geography": row["geography"],
            "category": row["category"],
            "annual_returns": annual_returns,
            "average_annual_return": round(np.mean(list(annual_returns.values())), 2) if annual_returns else 0,
            "total_return_5yr": round(((1 + returns_series).prod() - 1) * 100, 2) if len(returns_series) > 0 else 0
        }
    
    return results

def get_all_funds_annual_returns():
    """Get annual returns for all funds."""
    return get_fund_annual_returns()

