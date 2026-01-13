"""
Riskfolio optimization module for portfolio allocation.
Uses different optimization models based on risk buckets.
"""
import pandas as pd
import numpy as np
import riskfolio as rp
from dummy_data import get_funds_by_criteria, generate_dummy_funds

# Global fund data
FUND_DATA = None

def initialize_fund_data():
    """Initialize and cache fund data."""
    global FUND_DATA
    if FUND_DATA is None:
        FUND_DATA = generate_dummy_funds()
    return FUND_DATA

def risk_folio(
    currency,
    primary_risk_bucket,
    sub_risk_bucket,
    volatility_target_pct=None,
    drawdown_target_pct=None,
    fund_counts={},
    asset_split_targets={},
    geography_constraints={},
    tax_saver_target_pct=None
):
    """
    Optimize portfolio using riskfolio library.
    
    Parameters:
    -----------
    currency : str
        "INR" or "USD"
    primary_risk_bucket : str
        "LOW", "MEDIUM", or "HIGH"
    sub_risk_bucket : str
        e.g., "HIGH_HIGH", "HIGH_MEDIUM", "HIGH_LOW", etc.
    volatility_target_pct : float, optional
        Target volatility percentage
    drawdown_target_pct : float, optional
        Target maximum drawdown percentage
    fund_counts : dict
        Number of funds per category
    asset_split_targets : dict
        Target percentages for asset types
    geography_constraints : dict
        Geography weights (only for USD)
    tax_saver_target_pct : float, optional
        Tax saver percentage (only for INR)
    
    Returns:
    --------
    dict
        Recommended fund weights
    """
    # Initialize fund data
    df = initialize_fund_data()
    
    # Select funds based on currency
    available_funds = df[df["currency"] == currency].copy()
    
    # Filter by geography if USD
    if currency == "USD" and geography_constraints:
        # Get funds from specified geographies
        geographies = list(geography_constraints.keys())
        available_funds = available_funds[available_funds["geography"].isin(geographies)]
    
    # Select funds based on fund_counts
    selected_funds = []
    
    for category, count in fund_counts.items():
        if count > 0:
            category_funds = available_funds[available_funds["category"] == category]
            
            # For USD, consider geography constraints when selecting
            if currency == "USD" and geography_constraints:
                # Try to distribute across geographies
                geo_funds = {}
                for geo in geography_constraints.keys():
                    geo_funds[geo] = category_funds[category_funds["geography"] == geo]
                
                # Select funds proportionally
                selected = []
                for geo, weight in geography_constraints.items():
                    geo_count = max(1, int(count * weight / 100))
                    if len(geo_funds[geo]) > 0:
                        selected.extend(geo_funds[geo].head(geo_count).to_dict('records'))
                
                # If not enough, fill from any geography
                while len(selected) < count and len(category_funds) > len(selected):
                    remaining = category_funds[~category_funds["name"].isin([f["name"] for f in selected])]
                    if len(remaining) > 0:
                        selected.append(remaining.iloc[0].to_dict())
            else:
                # For INR or no geography constraints, just select top funds
                selected = category_funds.head(count).to_dict('records')
            
            selected_funds.extend(selected[:count])
    
    if len(selected_funds) == 0:
        return {"error": "No funds found matching criteria"}
    
    # Prepare returns matrix
    returns_data = []
    fund_names = []
    
    for fund in selected_funds:
        returns_series = fund["returns_series"]
        returns_data.append(returns_series.values)
        fund_names.append(fund["name"])
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(
        np.array(returns_data).T,
        columns=fund_names,
        index=selected_funds[0]["returns_series"].index
    )
    
    # Convert to annual returns (multiply by 252 trading days)
    returns_df = returns_df * 252
    
    # Initialize portfolio object
    port = rp.Portfolio(returns=returns_df)
    
    # Select optimization model based on risk bucket
    model = select_optimization_model(primary_risk_bucket, sub_risk_bucket)
    
    # Set risk-free rate
    rf = 0.03  # 3% risk-free rate
    
    # Run optimization
    try:
        if model == "max_sharpe":
            # Maximize Sharpe ratio
            w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=rf, hist=True)
        elif model == "min_volatility":
            # Minimize volatility
            w = port.optimization(model='Classic', rm='MV', obj='MinRisk', rf=rf, hist=True)
        elif model == "max_alpha":
            # Maximize alpha (excess return) - use Sharpe as proxy
            w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=rf, hist=True)
        elif model == "risk_parity":
            # Risk parity optimization
            w = port.rp_optimization(model='Classic', rm='MV', rf=rf, b=None, hist=True)
        elif model == "max_return":
            # Maximize returns
            w = port.optimization(model='Classic', rm='MV', obj='MaxRet', rf=rf, hist=True)
        else:
            # Default to max Sharpe
            w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=rf, hist=True)
        
        # Extract weights
        if w is not None:
            if hasattr(w, 'values'):
                weights_array = w.values.flatten()
            elif isinstance(w, pd.Series):
                weights_array = w.values
            elif isinstance(w, (list, np.ndarray)):
                weights_array = np.array(w).flatten()
            else:
                weights_array = np.array([1.0 / len(fund_names)] * len(fund_names))
        else:
            weights_array = np.array([1.0 / len(fund_names)] * len(fund_names))
        
        # Create weights dictionary
        weights_dict = {}
        for idx, weight in enumerate(weights_array):
            if idx < len(fund_names) and weight > 0.001:  # Only include weights > 0.1%
                weights_dict[fund_names[idx]] = round(float(weight) * 100, 2)
        
        # Normalize to 100%
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: round(v * 100 / total, 2) for k, v in weights_dict.items()}
        else:
            # Fallback to equal weights
            equal_weight = 100 / len(fund_names)
            weights_dict = {name: round(equal_weight, 2) for name in fund_names}
        
        # Apply constraints post-optimization (simplified approach)
        # Adjust weights to meet asset split targets if provided
        if asset_split_targets:
            adjusted_weights = apply_asset_split_constraints(
                weights_dict, selected_funds, asset_split_targets
            )
            weights_dict = adjusted_weights
        
        # Apply geography constraints for USD
        if currency == "USD" and geography_constraints:
            adjusted_weights = apply_geography_constraints(
                weights_dict, selected_funds, geography_constraints
            )
            weights_dict = adjusted_weights
        
        # Apply tax saver constraint for INR
        if currency == "INR" and tax_saver_target_pct and tax_saver_target_pct > 0:
            adjusted_weights = apply_tax_saver_constraint(
                weights_dict, selected_funds, tax_saver_target_pct
            )
            weights_dict = adjusted_weights
        
        return {
            "weights": weights_dict,
            "funds": selected_funds,
            "model_used": model,
            "total_weight": sum(weights_dict.values())
        }
    
    except Exception as e:
        # Fallback to equal weights if optimization fails
        equal_weight = 100 / len(selected_funds)
        weights_dict = {fund["name"]: round(equal_weight, 2) for fund in selected_funds}
        
        return {
            "weights": weights_dict,
            "funds": selected_funds,
            "model_used": "equal_weight_fallback",
            "error": str(e),
            "total_weight": 100
        }

def apply_asset_split_constraints(weights_dict, selected_funds, asset_split_targets):
    """Adjust weights to meet asset split targets."""
    # Group funds by type
    type_weights = {}
    for fund in selected_funds:
        fund_type = fund["type"]
        if fund_type == "taxsaver":
            fund_type = "tax_saver"
        if fund_type not in type_weights:
            type_weights[fund_type] = []
        type_weights[fund_type].append(fund["name"])
    
    # Calculate current weights by type
    current_splits = {}
    for asset_type, fund_list in type_weights.items():
        current_splits[asset_type] = sum(weights_dict.get(f, 0) for f in fund_list)
    
    # Adjust to meet targets
    adjusted_weights = weights_dict.copy()
    for asset_type, target_pct in asset_split_targets.items():
        if asset_type in type_weights:
            fund_list = type_weights[asset_type]
            current_total = current_splits.get(asset_type, 0)
            target_total = target_pct
            
            if len(fund_list) > 0 and current_total > 0:
                # Scale weights proportionally
                scale_factor = target_total / current_total if current_total > 0 else target_total / len(fund_list)
                for fund_name in fund_list:
                    if fund_name in adjusted_weights:
                        adjusted_weights[fund_name] = adjusted_weights[fund_name] * scale_factor
    
    # Normalize
    total = sum(adjusted_weights.values())
    if total > 0:
        adjusted_weights = {k: round(v * 100 / total, 2) for k, v in adjusted_weights.items()}
    
    return adjusted_weights

def apply_geography_constraints(weights_dict, selected_funds, geography_constraints):
    """Adjust weights to meet geography constraints."""
    # Group funds by geography
    geo_weights = {}
    for fund in selected_funds:
        geo = fund["geography"]
        if geo not in geo_weights:
            geo_weights[geo] = []
        geo_weights[geo].append(fund["name"])
    
    # Calculate current weights by geography
    current_splits = {}
    for geo, fund_list in geo_weights.items():
        current_splits[geo] = sum(weights_dict.get(f, 0) for f in fund_list)
    
    # Adjust to meet targets
    adjusted_weights = weights_dict.copy()
    for geo, target_pct in geography_constraints.items():
        if geo in geo_weights:
            fund_list = geo_weights[geo]
            current_total = current_splits.get(geo, 0)
            target_total = target_pct
            
            if len(fund_list) > 0 and current_total > 0:
                scale_factor = target_total / current_total if current_total > 0 else target_total / len(fund_list)
                for fund_name in fund_list:
                    if fund_name in adjusted_weights:
                        adjusted_weights[fund_name] = adjusted_weights[fund_name] * scale_factor
    
    # Normalize
    total = sum(adjusted_weights.values())
    if total > 0:
        adjusted_weights = {k: round(v * 100 / total, 2) for k, v in adjusted_weights.items()}
    
    return adjusted_weights

def apply_tax_saver_constraint(weights_dict, selected_funds, tax_saver_target_pct):
    """Adjust weights to meet tax saver target."""
    tax_saver_funds = [f["name"] for f in selected_funds if f["type"] == "taxsaver"]
    current_total = sum(weights_dict.get(f, 0) for f in tax_saver_funds)
    
    if len(tax_saver_funds) > 0 and current_total > 0:
        adjusted_weights = weights_dict.copy()
        scale_factor = tax_saver_target_pct / current_total if current_total > 0 else tax_saver_target_pct / len(tax_saver_funds)
        
        for fund_name in tax_saver_funds:
            if fund_name in adjusted_weights:
                adjusted_weights[fund_name] = adjusted_weights[fund_name] * scale_factor
        
        # Normalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: round(v * 100 / total, 2) for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    return weights_dict

def select_optimization_model(primary_risk_bucket, sub_risk_bucket):
    """
    Select optimization model based on risk bucket.
    
    Model mapping:
    - HIGH: Use models that maximize alpha/returns (max_sharpe, max_alpha, max_return)
    - MEDIUM: Use balanced models (max_sharpe, risk_parity)
    - LOW: Use models that minimize risk (min_volatility, risk_parity)
    """
    if primary_risk_bucket == "HIGH":
        if "HIGH" in sub_risk_bucket:
            return "max_return"  # Very aggressive - maximize returns
        elif "MEDIUM" in sub_risk_bucket:
            return "max_alpha"   # Aggressive - maximize alpha
        else:  # HIGH_LOW
            return "max_sharpe"  # Growth but cautious - maximize risk-adjusted returns
    
    elif primary_risk_bucket == "MEDIUM":
        if "HIGH" in sub_risk_bucket:
            return "max_sharpe"  # Medium-high - maximize risk-adjusted returns
        elif "MEDIUM" in sub_risk_bucket:
            return "risk_parity" # Balanced - risk parity
        else:  # MEDIUM_LOW
            return "risk_parity" # Medium-low - risk parity with slight tilt
    
    else:  # LOW
        if "HIGH" in sub_risk_bucket:
            return "risk_parity" # Low-high - risk parity
        elif "MEDIUM" in sub_risk_bucket:
            return "min_volatility" # Conservative - minimize volatility
        else:  # LOW_LOW
            return "min_volatility" # Very conservative - minimize volatility

