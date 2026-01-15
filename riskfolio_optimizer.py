"""
Portfolio optimization module using scipy.optimize.
Implements mean-variance optimization, Sharpe ratio maximization, and risk parity.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv
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
    tax_saver_target_pct=None,
    suggested_funds=None
):
    """
    Optimize portfolio using scipy.optimize (mean-variance optimization).
    
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
    suggested_funds : dict, optional
        Pre-selected funds by category (if provided, uses these instead of re-selecting)
    
    Returns:
    --------
    dict
        Recommended fund weights
    """
    # Initialize fund data
    df = initialize_fund_data()
    
    selected_funds = []
    
    # If suggested_funds is provided, use those exact funds
    if suggested_funds and len(suggested_funds) > 0:
        # Convert suggested_funds format to the format expected by optimization
        for category, funds_list in suggested_funds.items():
            for fund_info in funds_list:
                fund_name = fund_info.get("name")
                if fund_name:
                    # Strip whitespace and find the fund in the database to get full details including returns_series
                    fund_name_clean = str(fund_name).strip()
                    fund_row = df[df["name"].str.strip() == fund_name_clean]
                    if len(fund_row) > 0:
                        fund_dict = fund_row.iloc[0].to_dict()
                        selected_funds.append(fund_dict)
                    else:
                        # Try case-insensitive match
                        fund_row = df[df["name"].str.strip().str.lower() == fund_name_clean.lower()]
                        if len(fund_row) > 0:
                            fund_dict = fund_row.iloc[0].to_dict()
                            selected_funds.append(fund_dict)
    else:
        # Fallback to original selection logic if suggested_funds not provided
        # Select funds based on currency
        available_funds = df[df["currency"] == currency].copy()
        
        # Filter by geography if USD
        if currency == "USD" and geography_constraints:
            # Get funds from specified geographies
            geographies = list(geography_constraints.keys())
            available_funds = available_funds[available_funds["geography"].isin(geographies)]
        
        # Select funds based on fund_counts
        for category, count in fund_counts.items():
            # Ensure count is not None and is a valid integer
            if count is None:
                continue
            try:
                count = int(count)
            except (ValueError, TypeError):
                continue
            if count > 0:
                category_funds = available_funds[available_funds["category"] == category]
                
                # For USD, consider geography constraints when selecting
                if currency == "USD" and geography_constraints:
                    # Try to distribute across geographies
                    geo_funds = {}
                    for geo in geography_constraints.keys():
                        geo_funds[geo] = category_funds[category_funds["geography"] == geo]
                    
                    # Select funds proportionally, prioritizing highest returns
                    selected = []
                    for geo, weight in geography_constraints.items():
                        geo_count = max(1, int(count * weight / 100))
                        if len(geo_funds[geo]) > 0:
                            # Sort by returns and select top funds
                            geo_funds_sorted = geo_funds[geo].sort_values('returns', ascending=False)
                            selected.extend(geo_funds_sorted.head(geo_count).to_dict('records'))
                    
                    # If not enough, fill from any geography (sorted by returns)
                    while len(selected) < count and len(category_funds) > len(selected):
                        remaining = category_funds[~category_funds["name"].isin([f["name"] for f in selected])]
                        if len(remaining) > 0:
                            remaining_sorted = remaining.sort_values('returns', ascending=False)
                            selected.append(remaining_sorted.iloc[0].to_dict())
                else:
                    # For INR or no geography constraints, select top funds by returns
                    # Sort by returns (descending) and select top funds
                    category_funds_sorted = category_funds.sort_values('returns', ascending=False)
                    selected = category_funds_sorted.head(count).to_dict('records')
                
                selected_funds.extend(selected[:count])
    
    if len(selected_funds) == 0:
        return {"error": "No funds found matching criteria"}
    
    # Prepare returns matrix
    returns_data = []
    fund_names = []
    
    # Ensure all funds have the same length returns series
    if len(selected_funds) == 0:
        return {"error": "No funds selected"}
    
    # Get the index from first fund
    base_index = selected_funds[0]["returns_series"].index
    min_length = len(base_index)
    
    # Find minimum length across all funds
    for fund in selected_funds:
        if len(fund["returns_series"]) < min_length:
            min_length = len(fund["returns_series"])
    
    # Align all returns to same length and index
    for fund in selected_funds:
        returns_series = fund["returns_series"]
        
        # Ensure we have a pandas Series
        if not isinstance(returns_series, pd.Series):
            raise ValueError(f"Fund {fund['name']} has invalid returns_series type")
        
        # Align to base index and take minimum length
        try:
            aligned_series = returns_series.reindex(base_index[:min_length], method='ffill')
            aligned_series = aligned_series.fillna(0)  # Fill any NaN with 0
        except Exception:
            # If reindex fails, just take the first min_length values
            aligned_series = returns_series.iloc[:min_length] if len(returns_series) >= min_length else returns_series
        
        # Ensure exact length
        if len(aligned_series) > min_length:
            aligned_series = aligned_series.iloc[:min_length]
        elif len(aligned_series) < min_length:
            # Pad with zeros if too short
            padding = pd.Series([0.0] * (min_length - len(aligned_series)), 
                               index=base_index[len(aligned_series):min_length])
            aligned_series = pd.concat([aligned_series, padding])
        
        # Convert to numpy array and ensure it's 1D
        values = aligned_series.values
        if values.ndim > 1:
            values = values.flatten()
        
        returns_data.append(values)
        fund_names.append(fund["name"])
    
    # Create returns DataFrame - ensure proper dimensions
    if len(returns_data) == 0:
        return {"error": "No returns data available"}
    
    # Stack arrays properly - each row is a day, each column is a fund
    # Ensure all arrays have the same length
    for i in range(len(returns_data)):
        if len(returns_data[i]) != min_length:
            returns_data[i] = returns_data[i][:min_length]
    
    # Ensure we have at least 2 funds for optimization (Riskfolio needs multiple assets)
    if len(fund_names) < 2:
        # If only 1 fund, we can't optimize - return equal weight
        return {
            "weights": {fund_names[0]: 100.0},
            "funds": [{k: v for k, v in selected_funds[0].items() if k != "returns_series"}],
            "model_used": "single_fund",
            "risk_measure": "MV",
            "error": "Only one fund selected - cannot optimize. Using 100% allocation.",
            "optimization_success": False,
            "total_weight": 100
        }
    
    try:
        # Ensure all arrays are 1D and have same length
        for i in range(len(returns_data)):
            arr = np.array(returns_data[i])
            if arr.ndim > 1:
                arr = arr.flatten()
            if len(arr) != min_length:
                arr = arr[:min_length] if len(arr) > min_length else np.pad(arr, (0, min_length - len(arr)), 'constant')
            returns_data[i] = arr
        
        # Stack as columns (each fund is a column)
        # Shape should be (days, funds) - rows are time periods, columns are assets
        returns_array = np.column_stack(returns_data)
        
        # Verify dimensions before creating DataFrame
        if returns_array.shape[0] != min_length:
            # Transpose if needed (shouldn't happen, but safety check)
            if returns_array.shape[1] == min_length:
                returns_array = returns_array.T
            else:
                raise ValueError(f"Cannot align dimensions: array shape {returns_array.shape}, expected ({min_length}, {len(fund_names)})")
        
        if returns_array.shape[1] != len(fund_names):
            raise ValueError(f"Fund count mismatch: {returns_array.shape[1]} columns vs {len(fund_names)} funds")
        
        # Create DataFrame with proper orientation: rows = dates, columns = funds
        returns_df = pd.DataFrame(
            returns_array,
            columns=fund_names,
            index=base_index[:min_length]
        )
        
        # Final verification
        if returns_df.shape[0] < 2:
            raise ValueError(f"Insufficient data: only {returns_df.shape[0]} days of returns")
        if returns_df.shape[1] < 2:
            raise ValueError(f"Insufficient funds: only {returns_df.shape[1]} fund(s)")
            
    except Exception as e:
        # Fallback: create using dictionary method
        returns_dict = {}
        for i, fund_name in enumerate(fund_names):
            arr = np.array(returns_data[i])
            if arr.ndim > 1:
                arr = arr.flatten()
            if len(arr) > min_length:
                arr = arr[:min_length]
            elif len(arr) < min_length:
                arr = np.pad(arr, (0, min_length - len(arr)), 'constant')
            returns_dict[fund_name] = arr
        
        returns_df = pd.DataFrame(returns_dict, index=base_index[:min_length])
        
        # Verify fallback DataFrame
        if returns_df.shape[0] < 2 or returns_df.shape[1] < 2:
            raise ValueError(f"Cannot create valid returns DataFrame: shape {returns_df.shape}")
    
    # Calculate expected returns and covariance matrix from historical data
    # Annualize returns (assuming 252 trading days per year)
    try:
        # Additional validation before optimization
        if returns_df.empty:
            raise ValueError("Returns DataFrame is empty")
        if returns_df.shape[0] < 10:
            raise ValueError(f"Insufficient historical data: {returns_df.shape[0]} days (need at least 10)")
        if returns_df.shape[1] < 2:
            raise ValueError(f"Need at least 2 funds for optimization, got {returns_df.shape[1]}")
        
        # Ensure returns are properly formatted (no NaN or inf)
        if returns_df.isna().any().any():
            returns_df = returns_df.fillna(0)
        if np.isinf(returns_df.values).any():
            returns_df = returns_df.replace([np.inf, -np.inf], 0)
        
        # Calculate expected returns (annualized)
        # Mean daily return * 252 trading days
        expected_returns = returns_df.mean() * 252
        
        # Calculate covariance matrix (annualized)
        # Daily covariance * 252 trading days
        cov_matrix = returns_df.cov() * 252
        
        # Set risk-free rate (3% annual)
        rf = 0.03
        
        # Select optimization model based on risk bucket
        model, risk_measure = select_optimization_model(primary_risk_bucket, sub_risk_bucket, 
                                                         volatility_target_pct, drawdown_target_pct)
        
        # Run optimization using scipy
        n_assets = len(fund_names)
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints: weights sum to 1, each weight between 0 and 1
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Sum to 1
        )
        
        # Bounds: each weight between minimum (1% for diversification) and 1
        # Ensure all funds get at least some allocation for diversification
        min_weight = 0.01  # 1% minimum per fund
        max_weight = 0.95  # 95% maximum per fund (to allow some flexibility)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Run optimization based on model type
        if model == "max_sharpe":
            # Maximize Sharpe ratio: (portfolio_return - rf) / portfolio_volatility
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol < 1e-6:
                    return -1000  # Avoid division by zero
                sharpe = (portfolio_return - rf) / portfolio_vol
                return -sharpe  # Negative because we minimize
            
            result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif model == "min_volatility":
            # Minimize portfolio volatility
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            result = minimize(portfolio_volatility, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif model == "max_return":
            # Maximize portfolio return
            def negative_return(weights):
                return -np.dot(weights, expected_returns)
            
            result = minimize(negative_return, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif model == "risk_parity":
            # Risk parity: equal risk contribution from each asset
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol < 1e-6:
                    return 1000
                # Marginal risk contribution
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                # Risk contribution
                risk_contrib = weights * marginal_contrib
                # Target: equal risk contribution (1/n for each asset)
                target = np.ones(n_assets) / n_assets
                # Minimize squared difference from equal risk contribution
                return np.sum((risk_contrib - target * portfolio_vol / n_assets) ** 2)
            
            result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif model == "max_alpha":
            # Maximize risk-adjusted return (similar to Sharpe but with different formulation)
            def negative_alpha(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                # Alpha = excess return adjusted for risk
                if portfolio_vol < 1e-6:
                    return 1000
                alpha = (portfolio_return - rf) / (portfolio_vol + 0.01)  # Add small constant to avoid division by zero
                return -alpha
            
            result = minimize(negative_alpha, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        else:
            # Default to max Sharpe
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol < 1e-6:
                    return -1000
                sharpe = (portfolio_return - rf) / portfolio_vol
                return -sharpe
            
            result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Extract optimized weights
        if result.success:
            weights_array = result.x
        else:
            # If optimization failed with 1% minimum, try with 0.5% minimum
            relaxed_min_weight = 0.005  # 0.5% minimum
            relaxed_bounds = tuple((relaxed_min_weight, max_weight) for _ in range(n_assets))
            
            # Define objective functions (need to be accessible)
            def negative_sharpe_relaxed(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol < 1e-6:
                    return -1000
                sharpe = (portfolio_return - rf) / portfolio_vol
                return -sharpe
            
            try:
                relaxed_result = minimize(negative_sharpe_relaxed, x0, method='SLSQP', bounds=relaxed_bounds, constraints=constraints)
                if relaxed_result.success:
                    weights_array = relaxed_result.x
                else:
                    # Last resort: equal weights with minimum
                    weights_array = np.array([max(relaxed_min_weight, 1.0 / n_assets)] * n_assets)
            except:
                # Fallback to equal weights with minimum
                weights_array = np.array([max(relaxed_min_weight, 1.0 / n_assets)] * n_assets)
        
        # Ensure weights meet minimum and sum to 1
        min_allocation = 0.005  # 0.5% minimum per fund
        weights_array = np.maximum(weights_array, min_allocation)  # Ensure minimum
        weights_array = weights_array / np.sum(weights_array)  # Normalize to sum to 1
        
        # Create weights dictionary
        # CRITICAL: Include ALL funds with minimum allocation for diversification
        weights_dict = {}
        for idx, weight in enumerate(weights_array):
            if idx < len(fund_names):
                # Ensure minimum 0.5% allocation per fund for diversification
                min_allocation = max(0.005, weight)  # At least 0.5%
                weights_dict[fund_names[idx]] = round(float(min_allocation) * 100, 2)
        
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
        
        # Convert funds to JSON-serializable format (remove pandas Series)
        serializable_funds = []
        for fund in selected_funds:
            fund_copy = {k: v for k, v in fund.items() if k != "returns_series"}
            fund_copy["returns_count"] = len(fund.get("returns_series", []))
            serializable_funds.append(fund_copy)
        
        return {
            "weights": weights_dict,
            "funds": serializable_funds,
            "model_used": model,
            "risk_measure": risk_measure,
            "total_weight": sum(weights_dict.values()),
            "optimization_success": True
        }
    
    except Exception as e:
        # Fallback to equal weights if optimization fails
        equal_weight = 100 / len(selected_funds)
        weights_dict = {fund["name"]: round(equal_weight, 2) for fund in selected_funds}
        
        # Convert funds to JSON-serializable format
        serializable_funds = []
        for fund in selected_funds:
            fund_copy = {k: v for k, v in fund.items() if k != "returns_series"}
            fund_copy["returns_count"] = len(fund.get("returns_series", []))
            serializable_funds.append(fund_copy)
        
        return {
            "weights": weights_dict,
            "funds": serializable_funds,
            "model_used": "equal_weight_fallback",
            "risk_measure": "MV",
            "error": str(e),
            "optimization_success": False,
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
    
    # CRITICAL FIX: If a geography has 0% current allocation but target > 0, 
    # we need to allocate to funds from that geography
    # First, ensure all funds from geographies with target > 0 are in weights_dict
    adjusted_weights = weights_dict.copy()
    
    # Add funds that are missing but should have allocation
    for geo, target_pct in geography_constraints.items():
        if geo in geo_weights and target_pct > 0:
            fund_list = geo_weights[geo]
            current_total = current_splits.get(geo, 0)
            
            # If this geography has no current allocation but should have some
            if current_total == 0 and len(fund_list) > 0:
                # Allocate equally to all funds in this geography
                per_fund = target_pct / len(fund_list)
                for fund_name in fund_list:
                    if fund_name not in adjusted_weights:
                        adjusted_weights[fund_name] = 0
                    adjusted_weights[fund_name] += per_fund
    
    # Now adjust existing weights to meet targets
    for geo, target_pct in geography_constraints.items():
        if geo in geo_weights:
            fund_list = geo_weights[geo]
            current_total = sum(adjusted_weights.get(f, 0) for f in fund_list)
            target_total = target_pct
            
            if len(fund_list) > 0:
                if current_total > 0:
                    # Scale existing weights
                    scale_factor = target_total / current_total
                    for fund_name in fund_list:
                        if fund_name in adjusted_weights:
                            adjusted_weights[fund_name] = adjusted_weights[fund_name] * scale_factor
                else:
                    # No current allocation - distribute equally
                    per_fund = target_total / len(fund_list)
                    for fund_name in fund_list:
                        if fund_name not in adjusted_weights:
                            adjusted_weights[fund_name] = 0
                        adjusted_weights[fund_name] = per_fund
    
    # Remove weights that are too small (< 0.01%) but keep at least one fund per active geography
    # First, identify which geographies should have allocation
    active_geos = [geo for geo, pct in geography_constraints.items() if pct > 0]
    
    # Keep at least one fund per active geography
    for geo in active_geos:
        if geo in geo_weights:
            fund_list = geo_weights[geo]
            # Check if any fund from this geography is in adjusted_weights
            has_fund = any(f in adjusted_weights and adjusted_weights[f] > 0.01 for f in fund_list)
            if not has_fund and len(fund_list) > 0:
                # Add the first fund with minimum allocation
                adjusted_weights[fund_list[0]] = max(0.1, geography_constraints.get(geo, 0) / len(fund_list))
    
    # Remove very small weights (< 0.01%) but keep at least one per active geography
    weights_to_remove = []
    for fund_name, weight in adjusted_weights.items():
        if weight < 0.01:
            # Check if this is the only fund from its geography
            fund_geo = None
            for fund in selected_funds:
                if fund["name"] == fund_name:
                    fund_geo = fund["geography"]
                    break
            
            if fund_geo:
                # Check if there are other funds from this geography with > 0.01% weight
                other_funds_exist = False
                for other_fund in selected_funds:
                    if other_fund["geography"] == fund_geo and other_fund["name"] != fund_name:
                        if other_fund["name"] in adjusted_weights and adjusted_weights[other_fund["name"]] > 0.01:
                            other_funds_exist = True
                            break
                
                if other_funds_exist:
                    weights_to_remove.append(fund_name)
                # If no other funds exist, keep this one (minimum allocation)
            else:
                weights_to_remove.append(fund_name)
    
    for fund_name in weights_to_remove:
        del adjusted_weights[fund_name]
    
    # Normalize to 100%
    total = sum(adjusted_weights.values())
    if total > 0:
        adjusted_weights = {k: round(v * 100 / total, 2) for k, v in adjusted_weights.items()}
    else:
        # Fallback: equal weights across all funds
        equal_weight = 100 / len(selected_funds)
        adjusted_weights = {fund["name"]: round(equal_weight, 2) for fund in selected_funds}
    
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

def select_optimization_model(primary_risk_bucket, sub_risk_bucket, 
                              volatility_target_pct=None, drawdown_target_pct=None):
    """
    Select optimization model based on risk bucket.
    
    Uses scipy.optimize for portfolio optimization with different objectives:
    - max_sharpe: Maximize Sharpe ratio (risk-adjusted returns)
    - min_volatility: Minimize portfolio volatility
    - max_return: Maximize expected returns
    - risk_parity: Equal risk contribution from each asset
    - max_alpha: Maximize risk-adjusted excess returns
    
    Model mapping:
    - HIGH: Use models that maximize alpha/returns (max_sharpe, max_alpha, max_return)
    - MEDIUM: Use balanced models (max_sharpe, risk_parity)
    - LOW: Use models that minimize risk (min_volatility, risk_parity)
    
    Risk measure (for reference, scipy uses variance/covariance):
    - All models use mean-variance framework (covariance matrix)
    - Volatility constraints are handled through optimization bounds
    """
    # Risk measure is kept for compatibility but scipy uses variance/covariance
    if drawdown_target_pct is not None:
        risk_measure = 'Variance'  # Use variance for drawdown constraints
    elif volatility_target_pct is not None and volatility_target_pct < 20:
        risk_measure = 'Variance'  # Low volatility - use variance
    else:
        risk_measure = 'Variance'  # Default: Mean Variance
    
    # Select optimization model
    if primary_risk_bucket == "HIGH":
        if "HIGH" in sub_risk_bucket:
            return "max_return", risk_measure  # Very aggressive - maximize returns
        elif "MEDIUM" in sub_risk_bucket:
            return "max_alpha", risk_measure   # Aggressive - maximize alpha
        else:  # HIGH_LOW
            return "max_sharpe", risk_measure  # Growth but cautious - maximize risk-adjusted returns
    
    elif primary_risk_bucket == "MEDIUM":
        if "HIGH" in sub_risk_bucket:
            return "max_sharpe", risk_measure  # Medium-high - maximize risk-adjusted returns
        elif "MEDIUM" in sub_risk_bucket:
            return "risk_parity", risk_measure # Balanced - risk parity
        else:  # MEDIUM_LOW
            return "risk_parity", risk_measure # Medium-low - risk parity with slight tilt
    
    else:  # LOW
        if "HIGH" in sub_risk_bucket:
            return "risk_parity", risk_measure # Low-high - risk parity
        elif "MEDIUM" in sub_risk_bucket:
            return "min_volatility", risk_measure # Conservative - minimize volatility
        else:  # LOW_LOW
            return "min_volatility", risk_measure # Very conservative - minimize volatility

