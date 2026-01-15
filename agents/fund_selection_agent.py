"""
Fund Selection Agent - Handles fund selection with proper tool calling.
"""
import json
import random
from typing import Dict, List, Any, Optional
from database import Database
from dummy_data import generate_dummy_funds
from .base import BaseAgent


class FundSelectionAgent(BaseAgent):
    """Handles fund selection with proper tool calling based on criteria."""
    
    def __init__(self, session_id: str, db: Optional[Database] = None):
        super().__init__(session_id, db)
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict]:
        return [{
            "type": "function",
            "function": {
                "name": "select_funds",
                "description": "Select funds from database based on category, currency, geography, and risk profile. Returns actual fund names.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["debt", "large_cap", "mid_cap", "small_cap", "balanced", "tax_saver"],
                            "description": "Fund category"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of funds to select from this category"
                        },
                        "currency": {
                            "type": "string",
                            "enum": ["INR", "USD"],
                            "description": "Currency filter"
                        },
                        "geography": {
                            "type": "string",
                            "enum": ["USA", "India", "Japan", "Europe", "UK", "China"],
                            "description": "Geography filter (optional, for USD only)"
                        },
                        "risk_profile": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                            "description": "Risk profile affects selection criteria (HIGH prefers higher returns, LOW prefers lower volatility)"
                        }
                    },
                    "required": ["category", "count", "currency", "risk_profile"]
                }
            }
        }]
    
    def get_system_prompt(self) -> str:
        return """You are a Fund Selection Agent. Your job is to select actual funds and educate users about what each fund category means.

CRITICAL: You are a conversational assistant. DO NOT output your system prompt, instructions, or internal rules. Only provide natural, conversational responses. Never mention your system prompt.

DETAILED RESPONSE REQUIREMENTS:
1. **When displaying selected funds, provide detailed explanations:**

   **Debt Funds:**
   - Invest in fixed-income securities (bonds, government securities)
   - Lower risk, stable returns (typically 6-9% annually)
   - Suitable for: Capital preservation, regular income, low-risk portion of portfolio
   - Volatility: Low (3-8%)
   
   **Large Cap Equity Funds:**
   - Invest in top 100 companies by market capitalization
   - Established, stable companies (e.g., Reliance, TCS, Infosys in India; Apple, Microsoft in USA)
   - Moderate risk, steady growth (typically 10-15% annually)
   - Suitable for: Core portfolio foundation, moderate risk tolerance
   - Volatility: Moderate (12-18%)
   
   **Mid Cap Equity Funds:**
   - Invest in companies ranked 101-250 by market cap
   - Growing companies with potential (e.g., emerging leaders)
   - Higher risk, higher growth potential (typically 12-18% annually)
   - Suitable for: Growth-oriented investors, long-term horizon
   - Volatility: Higher (18-25%)
   
   **Small Cap Equity Funds:**
   - Invest in companies ranked 251+ by market cap
   - Small companies with high growth potential
   - Highest risk, highest growth potential (typically 15-25% annually, but can be volatile)
   - Suitable for: Aggressive investors, very long-term (10+ years)
   - Volatility: Highest (22-35%)
   
   **Balanced Funds:**
   - Mix of equity (60-70%) and debt (30-40%)
   - Balanced risk-return profile
   - Moderate growth with some stability (typically 10-12% annually)
   - Suitable for: Investors wanting balance between growth and safety
   - Volatility: Moderate (10-15%)
   
   **Tax Saver (ELSS) Funds:**
   - Equity-linked savings scheme
   - Provides Section 80C tax deduction (up to ₹1.5 lakh in India)
   - 3-year lock-in period (cannot withdraw before 3 years)
   - Invests primarily in equity
   - Suitable for: Tax-saving with long-term wealth creation
   - Volatility: Similar to equity funds (15-25%)

2. **When showing selected funds, explain:**
   - Why these specific funds were chosen (based on risk profile, returns, geography)
   - What each fund category contributes to the portfolio
   - How the allocation matches their risk profile
   - Expected portfolio characteristics

3. Use select_funds tool for EACH category in fund_counts
4. Pass: category, count, currency, risk_profile from context
5. For USD, also pass geography (distribute across geographies based on geography_constraints)
6. The tool returns actual fund names - use those exact names
7. Once all funds selected, show detailed fund table with explanations
8. **IMPORTANT**: After showing selected funds, ALWAYS ask the user if they want to proceed with portfolio optimization:
   - Explain what portfolio optimization means
   - Ask: "Shall I proceed with portfolio optimization?" or "Would you like me to proceed?"
   - Wait for user confirmation (yes, proceed, optimize, continue, etc.)
   - Only move to next agent (sub_risk_refinement) when user confirms

CRITICAL: 
- Always use actual fund names returned by the tool, never create generic names
- Provide educational context about what each fund category means
- Explain how the selected funds match their risk profile and goals
- **Show the actual fund names selected from the database based on geography, currency, and risk profile**
- **Explain that allocation percentages will be calculated during optimization based on these funds' metrics**
- ALWAYS ask for user confirmation before proceeding to optimization
- If user says yes/proceed/optimize/continue, then move to sub_risk_refinement agent"""

    def execute(self, user_message: str, context: Dict) -> Dict[str, Any]:
        # Check if funds already selected - if yes, check if user wants to proceed with optimization
        if context.get("suggested_funds") and len(context.get("suggested_funds", {})) > 0:
            # Check if user wants to proceed with optimization
            user_lower = user_message.lower().strip()
            proceed_keywords = ['yes', 'proceed', 'optimize', 'continue', 'go ahead', 'sure', 'okay', 'ok', "let's do it", 'optimise']
            
            if any(keyword in user_lower for keyword in proceed_keywords):
                return {
                    "response": "Perfect! I'll now refine your risk profile to get your exact volatility tolerance, and then proceed with portfolio optimization. Let me continue...",
                    "updated_context": context,
                    "next_agent": "sub_risk_refinement"
                }
            else:
                # User might have questions or wants to discuss funds
                return {
                    "response": "I'm here to help! Would you like to proceed with portfolio optimization, or do you have any questions about the selected funds? Say 'yes' or 'proceed' to continue with optimization.",
                    "updated_context": context,
                    "next_agent": None  # Stay in fund selection
                }
        
        # Check if we have required context
        if not context.get("currency") or not context.get("primary_risk_bucket") or not context.get("fund_counts"):
            return {
                "response": "Missing required context for fund selection.",
                "updated_context": context,
                "next_agent": None
            }
        
        # Select funds using the tool logic
        suggested_funds = self._select_funds_from_database(context)
        context["suggested_funds"] = suggested_funds
        
        # Save to database
        if self.db:
            try:
                self.db.save_user_preferences(self.session_id, context)
                self.db.save_suggested_funds(self.session_id, suggested_funds)
            except Exception as e:
                print(f"Database save error: {e}")
        
        # Format detailed response with fund counts and explanations
        response = "## 📊 Selected Funds for Your Portfolio\n\n"
        
        # Explain what fund selection means
        response += "I've selected specific mutual funds that match your risk profile, currency, and investment goals. "
        response += "These funds have been chosen based on their historical performance, risk characteristics, and alignment with your preferences.\n\n"
        
        # Show fund counts by category with detailed explanations
        fund_counts = context.get("fund_counts", {})
        primary_risk = context.get("primary_risk_bucket", "MEDIUM")
        
        category_detailed_explanations = {
            "debt": {
                "what": "Debt funds invest in fixed-income securities like bonds and government securities",
                "why": "Provides stability, capital preservation, and regular income",
                "returns": "Typically 6-9% annually",
                "risk": "Low risk, low volatility (3-8%)"
            },
            "large_cap": {
                "what": "Large cap funds invest in top 100 companies by market size (established leaders)",
                "why": "Foundation of portfolio - stable companies with steady growth",
                "returns": "Typically 10-15% annually",
                "risk": "Moderate risk, moderate volatility (12-18%)"
            },
            "mid_cap": {
                "what": "Mid cap funds invest in companies ranked 101-250 (growing companies)",
                "why": "Growth potential - companies with expansion opportunities",
                "returns": "Typically 12-18% annually",
                "risk": "Higher risk, higher volatility (18-25%)"
            },
            "small_cap": {
                "what": "Small cap funds invest in smaller companies (ranked 251+)",
                "why": "Highest growth potential - small companies that can become big",
                "returns": "Typically 15-25% annually (but can be very volatile)",
                "risk": "Highest risk, highest volatility (22-35%)"
            },
            "balanced": {
                "what": "Balanced funds mix equity (60-70%) and debt (30-40%)",
                "why": "Best of both worlds - growth with some stability",
                "returns": "Typically 10-12% annually",
                "risk": "Moderate risk, moderate volatility (10-15%)"
            },
            "tax_saver": {
                "what": "ELSS (Equity Linked Savings Scheme) funds for tax benefits",
                "why": "Section 80C tax deduction + long-term wealth creation",
                "returns": "Similar to equity funds (12-18% annually)",
                "risk": "Equity risk, 3-year lock-in period"
            }
        }
        
        response += "**Portfolio Structure & What Each Category Does:**\n\n"
        total_funds = 0
        for category, count in fund_counts.items():
            if count and count > 0:
                category_name = category.replace("_", " ").title()
                actual_count = len(suggested_funds.get(category, []))
                total_funds += actual_count
                cat_info = category_detailed_explanations.get(category, {})
                
                response += f"**{category_name}** ({actual_count} fund{'s' if actual_count != 1 else ''}):\n"
                response += f"- **What it is**: {cat_info.get('what', 'Diversified investment')}\n"
                response += f"- **Why it's included**: {cat_info.get('why', 'Portfolio diversification')}\n"
                response += f"- **Expected returns**: {cat_info.get('returns', 'Varies')}\n"
                response += f"- **Risk level**: {cat_info.get('risk', 'Moderate')}\n"
                response += "\n"
        
        response += f"**Total Funds Selected**: {total_funds} funds across {len([c for c, v in fund_counts.items() if v and v > 0])} categories\n\n"
        
        # Show geography distribution if USD
        currency = context.get("currency")
        if currency == "USD" and context.get("geography_constraints"):
            geography_distribution = {}
            for category, funds in suggested_funds.items():
                for fund in funds:
                    geo = fund.get("geography", "N/A")
                    geography_distribution[geo] = geography_distribution.get(geo, 0) + 1
            
            if geography_distribution:
                response += "**Geography Distribution (Number of Funds by Geography):**\n\n"
                geo_flags = {
                    'USA': '🇺🇸', 'India': '🇮🇳', 'Japan': '🇯🇵',
                    'Europe': '🇪🇺', 'UK': '🇬🇧', 'China': '🇨🇳'
                }
                for geo, count in sorted(geography_distribution.items(), key=lambda x: x[1], reverse=True):
                    flag = geo_flags.get(geo, '🌍')
                    geo_pct = context.get("geography_constraints", {}).get(geo, 0)
                    response += f"- {flag} **{geo}**: {count} fund{'s' if count != 1 else ''} (target allocation: {geo_pct}%)\n"
                response += "\n"
        
        # Explain how selection matches risk profile
        response += f"**How This Matches Your {primary_risk} Risk Profile:**\n"
        if primary_risk == "LOW":
            response += "Your portfolio emphasizes stability with more debt and large-cap funds. "
            response += "This provides capital preservation while still allowing for some growth through equity exposure.\n\n"
        elif primary_risk == "MEDIUM":
            response += "Your portfolio is balanced with a mix of equity and debt. "
            response += "This provides moderate growth potential while maintaining some stability through debt funds.\n\n"
        else:  # HIGH
            response += "Your portfolio focuses on growth with more mid-cap and small-cap funds. "
            response += "This provides higher return potential but comes with higher volatility. "
            response += "Suitable for long-term investors comfortable with market swings.\n\n"
        
        response += "**Selected Funds Details:**\n\n"
        response += self._format_fund_table(suggested_funds)
        
        # Add prompt asking if user wants to proceed with optimization
        response += "\n---\n\n"
        response += "## 🚀 Ready for Portfolio Optimization?\n\n"
        response += "Great! I've selected the **actual fund names** for your portfolio based on:\n"
        response += f"- Your risk profile ({primary_risk})\n"
        response += f"- Currency ({currency})\n"
        if currency == "USD" and context.get("geography_constraints"):
            response += "- Your geography preferences (country allocation you confirmed)\n"
        response += "- Fund performance metrics (returns, volatility, drawdown)\n\n"
        response += "**What Happens Next?**\n"
        response += "1. **Sub-Risk Refinement**: I'll refine your risk profile to get your exact volatility/drawdown tolerance\n"
        response += "2. **Portfolio Optimization**: I'll calculate the optimal weight (percentage allocation) for each selected fund using:\n"
        response += "   - The selected funds' historical performance metrics\n"
        response += "   - Your risk tolerance and volatility targets\n"
        response += "   - Advanced mathematical models to maximize returns while staying within your risk limits\n"
        response += "   - Geography and category constraints\n\n"
        response += "**The optimization will determine:**\n"
        response += "- What percentage of your portfolio should go to each fund\n"
        response += "- How to balance risk and return based on the actual fund metrics\n"
        response += "- The final allocation that matches your investment goals\n\n"
        response += "**Would you like me to proceed?** Say **'yes'**, **'proceed'**, **'optimize'**, or **'continue'** to start the optimization process. "
        response += "Or if you have any questions about the selected funds, feel free to ask!"
        
        return {
            "response": response,
            "updated_context": context,
            "next_agent": None  # Stay in fund selection until user confirms
        }
    
    def _select_funds_from_database(self, context: Dict) -> Dict:
        """Select funds based on criteria with geography-based distribution."""
        df = generate_dummy_funds()
        currency = context.get("currency")
        fund_counts = context.get("fund_counts", {})
        geography_constraints = context.get("geography_constraints", {})
        primary_risk = context.get("primary_risk_bucket", "MEDIUM")
        
        # Filter by currency
        df = df[df["currency"] == currency]
        
        suggested = {}
        random.seed(hash(self.session_id) % 2**32)
        
        # Calculate total funds needed
        total_funds_needed = sum(int(count) for count in fund_counts.values() if count and count > 0)
        
        # For USD, distribute funds across geographies based on geography_constraints
        if currency == "USD" and geography_constraints and total_funds_needed > 0:
            # Step 1: compute raw fractional allocations per geography
            raw_allocations = {}
            for geo, pct in geography_constraints.items():
                if pct and pct > 0:
                    raw = total_funds_needed * (pct / 100.0)
                    raw_allocations[geo] = raw
            
            # Step 2: take floors and track remainders
            geography_fund_allocation = {}
            remainders = []
            allocated = 0
            for geo, raw in raw_allocations.items():
                base = int(raw)
                geography_fund_allocation[geo] = base
                allocated += base
                remainders.append((geo, raw - base))
            
            # Step 3: distribute remaining funds based on largest remainders
            remaining = max(0, total_funds_needed - allocated)
            if remaining > 0 and remainders:
                remainders.sort(key=lambda x: x[1], reverse=True)
                idx = 0
                while remaining > 0:
                    geo = remainders[idx][0]
                    geography_fund_allocation[geo] = geography_fund_allocation.get(geo, 0) + 1
                    remaining -= 1
                    idx = (idx + 1) % len(remainders)
            
            # Final safety: if we somehow overshot, trim from smallest allocations
            total_allocated = sum(geography_fund_allocation.values())
            if total_allocated > total_funds_needed:
                for geo, _ in sorted(geography_fund_allocation.items(), key=lambda x: x[1]):
                    if total_allocated <= total_funds_needed:
                        break
                    if geography_fund_allocation[geo] > 0:
                        geography_fund_allocation[geo] -= 1
                        total_allocated -= 1
            
            # Distribute funds by category across geographies
            for category, count in fund_counts.items():
                if count and count > 0:
                    category_funds = df[df["category"] == category].copy()
                    
                    # Filter by available geographies
                    available_geos = [g for g in geography_fund_allocation.keys() if geography_fund_allocation[g] > 0]
                    category_funds = category_funds[category_funds["geography"].isin(available_geos)]
                    
                    if len(category_funds) > 0:
                        # Apply risk-based filtering
                        if primary_risk == "HIGH":
                            category_funds = category_funds.sort_values('returns', ascending=False)
                            top_n = max(1, int(len(category_funds) * 0.7))
                            pool = category_funds.head(top_n)
                        elif primary_risk == "LOW":
                            category_funds = category_funds.sort_values('volatility', ascending=True)
                            top_n = max(1, int(len(category_funds) * 0.7))
                            pool = category_funds.head(top_n)
                        else:  # MEDIUM
                            median_return = category_funds['returns'].median()
                            pool = category_funds[
                                (category_funds['returns'] >= median_return - 3) & 
                                (category_funds['returns'] <= median_return + 3)
                            ]
                            if len(pool) == 0:
                                pool = category_funds
                        
                        # Distribute count across geographies proportionally
                        fund_list = []
                        remaining_count = int(count)
                        
                        # Sort geographies by allocation percentage (descending)
                        geo_order = sorted(geography_fund_allocation.items(), key=lambda x: x[1], reverse=True)
                        
                        for geo, geo_fund_count in geo_order:
                            if remaining_count <= 0:
                                break
                            
                            # Calculate how many funds from this geography for this category
                            # Distribute proportionally based on geography allocation
                            geo_pct = geography_constraints.get(geo, 0)
                            if geo_pct > 0:
                                # Calculate proportional allocation for this category
                                num_from_geo = max(1, round(int(count) * geo_pct / 100))
                                num_from_geo = min(num_from_geo, remaining_count)
                                
                                # Get funds from this geography and category
                                geo_pool = pool[pool["geography"] == geo].copy()
                                
                                if len(geo_pool) > 0:
                                    num_to_select = min(num_from_geo, len(geo_pool), remaining_count)
                                    if num_to_select > 0:
                                        if len(geo_pool) > num_to_select:
                                            selected = geo_pool.sample(n=num_to_select, random_state=random.randint(0, 2**32))
                                        else:
                                            selected = geo_pool
                                        
                                        for _, row in selected.iterrows():
                                            fund_list.append({
                                                "name": str(row["name"]).strip(),
                                                "returns": float(row["returns"]),
                                                "volatility": float(row["volatility"]),
                                                "max_drawdown": float(row.get("max_drawdown", 0)),
                                                "geography": str(row.get("geography", "N/A")),
                                                "type": str(row.get("type", "N/A")),
                                                "category": category
                                            })
                                        remaining_count -= num_to_select
                        
                        # If we still need more funds, fill from any available geography
                        if remaining_count > 0:
                            remaining_pool = pool[~pool["geography"].isin([f.get("geography") for f in fund_list])]
                            if len(remaining_pool) == 0:
                                remaining_pool = pool
                            
                            if len(remaining_pool) > 0:
                                num_to_select = min(remaining_count, len(remaining_pool))
                                if num_to_select > 0:
                                    if len(remaining_pool) > num_to_select:
                                        selected = remaining_pool.sample(n=num_to_select, random_state=random.randint(0, 2**32))
                                    else:
                                        selected = remaining_pool
                                    
                                    for _, row in selected.iterrows():
                                        fund_list.append({
                                            "name": str(row["name"]).strip(),
                                            "returns": float(row["returns"]),
                                            "volatility": float(row["volatility"]),
                                            "max_drawdown": float(row.get("max_drawdown", 0)),
                                            "geography": str(row.get("geography", "N/A")),
                                            "type": str(row.get("type", "N/A")),
                                            "category": category
                                        })
                        
                        if len(fund_list) > 0:
                            suggested[category] = fund_list
        else:
            # For INR or no geography constraints, use simple selection
            for category, count in fund_counts.items():
                if count and count > 0:
                    category_funds = df[df["category"] == category].copy()
                    
                    if len(category_funds) > 0:
                        # Apply risk-based filtering
                        if primary_risk == "HIGH":
                            category_funds = category_funds.sort_values('returns', ascending=False)
                            top_n = max(1, int(len(category_funds) * 0.7))
                            pool = category_funds.head(top_n)
                        elif primary_risk == "LOW":
                            category_funds = category_funds.sort_values('volatility', ascending=True)
                            top_n = max(1, int(len(category_funds) * 0.7))
                            pool = category_funds.head(top_n)
                        else:  # MEDIUM
                            median_return = category_funds['returns'].median()
                            pool = category_funds[
                                (category_funds['returns'] >= median_return - 3) & 
                                (category_funds['returns'] <= median_return + 3)
                            ]
                            if len(pool) == 0:
                                pool = category_funds
                        
                        # Randomly select
                        num_to_select = min(int(count), len(pool))
                        if num_to_select > 0:
                            if len(pool) > num_to_select:
                                selected = pool.sample(n=num_to_select, random_state=random.randint(0, 2**32))
                            else:
                                selected = pool
                            
                            fund_list = []
                            for _, row in selected.iterrows():
                                fund_list.append({
                                    "name": str(row["name"]).strip(),
                                    "returns": float(row["returns"]),
                                    "volatility": float(row["volatility"]),
                                    "max_drawdown": float(row.get("max_drawdown", 0)),
                                    "geography": str(row.get("geography", "N/A")),
                                    "type": str(row.get("type", "N/A")),
                                    "category": category
                                })
                            
                            if len(fund_list) > 0:
                                suggested[category] = fund_list
        
        return suggested
    
    def _format_fund_table(self, suggested_funds: Dict) -> str:
        """Format funds in a markdown table with proper formatting."""
        response = "| Fund Name | Returns | Volatility | Max Drawdown | Geography |\n"
        response += "|-----------|---------|------------|--------------|----------|\n"
        
        geo_flags = {
            'USA': '🇺🇸', 'India': '🇮🇳', 'Japan': '🇯🇵',
            'Europe': '🇪🇺', 'UK': '🇬🇧', 'China': '🇨🇳'
        }
        
        for category, funds in suggested_funds.items():
            if funds:
                for fund in funds:
                    geo = fund.get('geography', '')
                    flag = geo_flags.get(geo, '') if geo else ''
                    geo_display = f"{flag} {geo}" if geo else "N/A"
                    
                    # Format drawdown to 2 decimal places
                    max_drawdown = fund.get('max_drawdown', 0)
                    if isinstance(max_drawdown, (int, float)):
                        max_drawdown = f"{max_drawdown:.2f}"
                    else:
                        max_drawdown = str(max_drawdown)
                    
                    # Format returns and volatility
                    returns = f"{fund.get('returns', 0):.2f}" if isinstance(fund.get('returns'), (int, float)) else str(fund.get('returns', 0))
                    volatility = f"{fund.get('volatility', 0):.2f}" if isinstance(fund.get('volatility'), (int, float)) else str(fund.get('volatility', 0))
                    
                    response += f"| **{fund['name']}** | **{returns}%** | {volatility}% | {max_drawdown}% | {geo_display} |\n"
        
        return response

