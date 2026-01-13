"""
Agentic AI Chatbot with autonomous decision-making and tool usage.
Uses an agnostic agentic framework pattern.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import Dict, List, Any, Optional
from riskfolio_optimizer import risk_folio

load_dotenv()

class AgenticPortfolioAgent:
    """
    Agentic AI agent that autonomously manages portfolio recommendation conversations.
    Uses tools and makes decisions independently.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.state = {
            "step": 0,
            "currency": None,
            "primary_risk_bucket": None,
            "sub_risk_bucket": None,
            "volatility_target_pct": None,
            "drawdown_target_pct": None,
            "fund_counts": {},
            "asset_split_targets": {},
            "geography_constraints": {},
            "tax_saver_target_pct": None,
            "conversation_history": []
        }
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict]:
        """Define tools available to the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "update_currency",
                    "description": "Update the investment currency (INR or USD)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "currency": {
                                "type": "string",
                                "enum": ["INR", "USD"],
                                "description": "The currency for investment"
                            }
                        },
                        "required": ["currency"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_risk_profile",
                    "description": "Update the risk profile based on user responses",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "primary_risk": {
                                "type": "string",
                                "enum": ["LOW", "MEDIUM", "HIGH"],
                                "description": "Primary risk bucket"
                            },
                            "sub_risk": {
                                "type": "string",
                                "enum": ["LOW_LOW", "LOW_MEDIUM", "LOW_HIGH", 
                                        "MEDIUM_LOW", "MEDIUM_MEDIUM", "MEDIUM_HIGH",
                                        "HIGH_LOW", "HIGH_MEDIUM", "HIGH_HIGH"],
                                "description": "Sub-risk bucket"
                            },
                            "volatility_target": {
                                "type": "number",
                                "description": "Target volatility percentage (optional)"
                            },
                            "drawdown_target": {
                                "type": "number",
                                "description": "Target maximum drawdown percentage (optional)"
                            }
                        },
                        "required": ["primary_risk", "sub_risk"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_asset_split",
                    "description": "Update asset allocation targets",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "debt": {"type": "number"},
                            "equity": {"type": "number"},
                            "balanced": {"type": "number"},
                            "tax_saver": {"type": "number"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_geography",
                    "description": "Update geography constraints for USD investments",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "USA": {"type": "number"},
                            "India": {"type": "number"},
                            "Japan": {"type": "number"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_portfolio",
                    "description": "Generate optimized portfolio weights using risk_folio function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confirm": {
                                "type": "boolean",
                                "description": "Whether to proceed with optimization"
                            }
                        },
                        "required": ["confirm"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool call."""
        if tool_name == "update_currency":
            self.state["currency"] = arguments["currency"]
            self.state["step"] = 1
            return f"Currency set to {arguments['currency']}"
        
        elif tool_name == "update_risk_profile":
            self.state["primary_risk_bucket"] = arguments["primary_risk"]
            self.state["sub_risk_bucket"] = arguments["sub_risk"]
            if "volatility_target" in arguments:
                self.state["volatility_target_pct"] = arguments["volatility_target"]
            if "drawdown_target" in arguments:
                self.state["drawdown_target_pct"] = arguments["drawdown_target"]
            
            # Set default fund counts
            self._set_default_fund_counts()
            return f"Risk profile updated: {arguments['primary_risk']} - {arguments['sub_risk']}"
        
        elif tool_name == "update_asset_split":
            if not self.state["asset_split_targets"]:
                self.state["asset_split_targets"] = {}
            self.state["asset_split_targets"].update(arguments)
            return "Asset split targets updated"
        
        elif tool_name == "update_geography":
            if not self.state["geography_constraints"]:
                self.state["geography_constraints"] = {}
            self.state["geography_constraints"].update(arguments)
            return "Geography constraints updated"
        
        elif tool_name == "optimize_portfolio":
            if arguments["confirm"]:
                payload = self._build_optimization_payload()
                result = risk_folio(**payload)
                return result
            return "Optimization cancelled"
        
        return "Tool executed"
    
    def _set_default_fund_counts(self):
        """Set default fund counts based on risk profile."""
        if self.state["primary_risk_bucket"] == "LOW":
            self.state["fund_counts"] = {
                "debt": 1, "large_cap": 2, "mid_cap": 1, 
                "small_cap": 0, "balanced": 1, "tax_saver": 0
            }
        elif self.state["primary_risk_bucket"] == "MEDIUM":
            self.state["fund_counts"] = {
                "debt": 1, "large_cap": 1, "mid_cap": 1,
                "small_cap": 1, "balanced": 1, "tax_saver": 0
            }
        elif self.state["primary_risk_bucket"] == "HIGH":
            self.state["fund_counts"] = {
                "debt": 0, "large_cap": 1, "mid_cap": 2,
                "small_cap": 2, "balanced": 1, "tax_saver": 0
            }
    
    def _build_optimization_payload(self) -> Dict:
        """Build optimization payload from current state."""
        # Set defaults if not set
        if not self.state["sub_risk_bucket"] and self.state["primary_risk_bucket"]:
            if self.state["primary_risk_bucket"] == "HIGH":
                self.state["sub_risk_bucket"] = "HIGH_MEDIUM"
            elif self.state["primary_risk_bucket"] == "MEDIUM":
                self.state["sub_risk_bucket"] = "MEDIUM_MEDIUM"
            else:
                self.state["sub_risk_bucket"] = "LOW_MEDIUM"
        
        if not self.state["volatility_target_pct"] and not self.state["drawdown_target_pct"]:
            sub_risk = self.state["sub_risk_bucket"]
            defaults = {
                "HIGH_HIGH": 50, "HIGH_MEDIUM": 40, "HIGH_LOW": 30,
                "MEDIUM_HIGH": 30, "MEDIUM_MEDIUM": 25, "MEDIUM_LOW": 20,
                "LOW_HIGH": 20, "LOW_MEDIUM": 15, "LOW_LOW": 10
            }
            self.state["volatility_target_pct"] = defaults.get(sub_risk, 25)
        
        if not self.state["asset_split_targets"]:
            if self.state["currency"] == "USD":
                if self.state["primary_risk_bucket"] == "LOW":
                    self.state["asset_split_targets"] = {"debt": 40, "equity": 40, "balanced": 20}
                elif self.state["primary_risk_bucket"] == "MEDIUM":
                    self.state["asset_split_targets"] = {"debt": 25, "equity": 55, "balanced": 20}
                else:
                    self.state["asset_split_targets"] = {"debt": 10, "equity": 75, "balanced": 15}
            else:
                if self.state["primary_risk_bucket"] == "LOW":
                    self.state["asset_split_targets"] = {"debt": 40, "equity": 40, "balanced": 20, "tax_saver": 0}
                elif self.state["primary_risk_bucket"] == "MEDIUM":
                    self.state["asset_split_targets"] = {"debt": 25, "equity": 55, "balanced": 20, "tax_saver": 0}
                else:
                    self.state["asset_split_targets"] = {"debt": 10, "equity": 75, "balanced": 15, "tax_saver": 0}
        
        if self.state["currency"] == "USD" and not self.state["geography_constraints"]:
            self.state["geography_constraints"] = {"USA": 70, "India": 20, "Japan": 10}
        
        return {
            "currency": self.state["currency"],
            "primary_risk_bucket": self.state["primary_risk_bucket"],
            "sub_risk_bucket": self.state["sub_risk_bucket"],
            "volatility_target_pct": self.state["volatility_target_pct"],
            "drawdown_target_pct": self.state["drawdown_target_pct"],
            "fund_counts": self.state["fund_counts"],
            "asset_split_targets": self.state["asset_split_targets"],
            "geography_constraints": self.state["geography_constraints"],
            "tax_saver_target_pct": self.state["tax_saver_target_pct"] or 0
        }
    
    def get_system_prompt(self) -> str:
        """Get system prompt for the agentic AI."""
        return """You are an autonomous AI agent that helps users build diversified mutual fund portfolios.
You have access to tools that let you:
1. Update currency (INR/USD)
2. Update risk profile (LOW/MEDIUM/HIGH with sub-buckets)
3. Update asset allocation targets
4. Update geography constraints (for USD)
5. Optimize portfolios

You must:
- Infer risk tolerance from user responses (don't ask directly)
- Ask clarifying questions when needed
- Use tools autonomously when you have enough information
- Explain your decisions to the user
- Keep conversations natural and friendly

CONVERSATION FLOW:
1. Ask about investment goals
2. Determine currency (INR/USD)
3. Infer risk from scenario questions
4. Refine sub-risk and get volatility/drawdown targets
5. Get asset split preferences (optional)
6. Confirm and optimize when ready

Use tools proactively when you have sufficient information."""
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process user message with agentic decision-making."""
        self.state["conversation_history"].append({"role": "user", "content": user_message})
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "system", "content": f"Current state: {json.dumps(self.state, default=str)}"}
        ] + self.state["conversation_history"]
        
        try:
            # Call with function calling enabled
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini (agnostic model)
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            message = response.choices[0].message
            tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else []
            
            # Handle tool calls
            tool_results = []
            optimization_result = None
            
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        result = self._execute_tool(tool_name, arguments)
                        
                        if tool_name == "optimize_portfolio" and isinstance(result, dict):
                            optimization_result = result
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": json.dumps({"status": "optimization_complete", "result": result})
                            })
                        else:
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": str(result)
                            })
                    except Exception as e:
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": f"Error: {str(e)}"
                        })
                
                # Add tool results and get final response
                messages.append(message)
                messages.extend(tool_results)
                
                final_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                
                bot_message = final_response.choices[0].message.content
            else:
                bot_message = message.content
            
            self.state["conversation_history"].append({"role": "assistant", "content": bot_message})
            
            return {
                "response": bot_message,
                "state": self.state,
                "optimization_result": optimization_result
            }
        
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                "state": self.state,
                "optimization_result": None
            }
    
    def reset(self):
        """Reset agent state."""
        self.state = {
            "step": 0,
            "currency": None,
            "primary_risk_bucket": None,
            "sub_risk_bucket": None,
            "volatility_target_pct": None,
            "drawdown_target_pct": None,
            "fund_counts": {},
            "asset_split_targets": {},
            "geography_constraints": {},
            "tax_saver_target_pct": None,
            "conversation_history": []
        }

# Backward compatibility
PortfolioChatbot = AgenticPortfolioAgent

