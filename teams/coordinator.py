"""
Coordinator Agent - Main orchestrator that routes to appropriate sub-agents.
"""
from typing import Dict, Any
from database import Database
from agents import (
    CurrencyAgent,
    GeographyAgent,
    RiskAssessmentAgent,
    FundSelectionAgent,
    SubRiskRefinementAgent,
    OptimizationAgent
)


class CoordinatorAgent:
    """Main orchestrator that routes to appropriate sub-agents."""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        try:
            self.db = Database()
        except Exception as e:
            print(f"Database initialization warning: {e}")
            self.db = None
        
        # Initialize all sub-agents
        self.currency_agent = CurrencyAgent(session_id, self.db)
        self.geography_agent = GeographyAgent(session_id, self.db)
        self.risk_agent = RiskAssessmentAgent(session_id, self.db)
        self.fund_selection_agent = FundSelectionAgent(session_id, self.db)
        self.sub_risk_agent = SubRiskRefinementAgent(session_id, self.db)
        self.optimization_agent = OptimizationAgent(session_id, self.db)
        
        # Current agent
        self.current_agent = "currency"
        
        # Context (shared state)
        self.context = {
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
            "suggested_funds": {},
            "conversation_history": []
        }
        
        # Load from database
        self._load_context_from_database()
    
    def _load_context_from_database(self):
        """Load context from database if available."""
        if self.db:
            try:
                db_prefs = self.db.get_user_preferences(self.session_id)
                if db_prefs:
                    for key in ["currency", "primary_risk_bucket", "sub_risk_bucket",
                               "volatility_target_pct", "drawdown_target_pct",
                               "fund_counts", "asset_split_targets", "geography_constraints",
                               "tax_saver_target_pct"]:
                        if self.context.get(key) is None and db_prefs.get(key) is not None:
                            self.context[key] = db_prefs[key]
                
                db_funds = self.db.get_suggested_funds(self.session_id)
                if db_funds:
                    self.context["suggested_funds"] = db_funds
                
                # Load conversation history from database
                db_history = self.db.get_conversation_history(self.session_id, limit=100)
                if db_history:
                    # Convert database format to conversation history format
                    self.context["conversation_history"] = []
                    for conv in db_history:
                        if conv.get("user_message"):
                            self.context["conversation_history"].append({
                                "role": "user",
                                "content": conv["user_message"]
                            })
                        if conv.get("bot_response"):
                            self.context["conversation_history"].append({
                                "role": "assistant",
                                "content": conv["bot_response"]
                            })
                
                # Determine current agent based on context
                self._determine_current_agent()
            except Exception as e:
                print(f"Error loading context: {e}")
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process user message through appropriate agent with enhanced context memory."""
        # Load context from database at the start of each chat to ensure we have latest state
        self._load_context_from_database()
        
        # Add to conversation history
        self.context["conversation_history"].append({"role": "user", "content": user_message})
        
        # Update all agents' conversation history for better context
        for agent in [self.currency_agent, self.geography_agent, self.risk_agent, 
                      self.fund_selection_agent, self.sub_risk_agent, self.optimization_agent]:
            agent.conversation_history = self.context["conversation_history"].copy()
        
        # Determine current agent based on context (prevents loops)
        self._determine_current_agent()
        
        # Route to appropriate agent
        agent_map = {
            "currency": self.currency_agent,
            "geography": self.geography_agent,
            "risk_assessment": self.risk_agent,
            "fund_selection": self.fund_selection_agent,
            "sub_risk_refinement": self.sub_risk_agent,
            "optimization": self.optimization_agent
        }
        
        current_agent_obj = agent_map.get(self.current_agent)
        if not current_agent_obj:
            self.current_agent = "currency"
            current_agent_obj = self.currency_agent
        
        # Execute agent
        result = current_agent_obj.execute(user_message, self.context)
        
        # Update context
        self.context.update(result.get("updated_context", {}))
        
        # Move to next agent if specified
        if result.get("next_agent"):
            next_agent = result["next_agent"]
            # Only move if we're actually moving to a different agent
            if self.current_agent != next_agent:
                self.current_agent = next_agent
                # Mark that we've moved to prevent duplicate responses
                self.context[f"_moved_to_{next_agent}"] = True
                
                # If moving to fund_selection, execute it immediately (automatic progression)
                if self.current_agent == "fund_selection" and not self.context.get("suggested_funds"):
                    fund_result = self.fund_selection_agent.execute("", self.context)
                    self.context.update(fund_result.get("updated_context", {}))
                    if fund_result.get("response"):
                        result["response"] = (result.get("response", "") + "\n\n" + fund_result["response"]).strip()
                    if fund_result.get("next_agent"):
                        self.current_agent = fund_result["next_agent"]
                        self.context[f"_moved_to_{fund_result['next_agent']}"] = True
        else:
            # Re-determine agent based on updated context
            self._determine_current_agent()
        
        # Add bot response to history
        if result.get("response"):
            self.context["conversation_history"].append({"role": "assistant", "content": result["response"]})
        
        # Update all agents' conversation history
        for agent in [self.currency_agent, self.geography_agent, self.risk_agent, 
                      self.fund_selection_agent, self.sub_risk_agent, self.optimization_agent]:
            agent.conversation_history = self.context["conversation_history"].copy()
        
        # Save to database
        if self.db and result.get("response"):
            try:
                self.db.save_conversation(self.session_id, user_message, result["response"], self.context)
                self.db.save_user_preferences(self.session_id, self.context)
                if self.context.get("suggested_funds"):
                    self.db.save_suggested_funds(self.session_id, self.context["suggested_funds"])
            except Exception as e:
                print(f"Error saving conversation: {e}")
        
        return {
            "response": result.get("response", ""),
            "state": self.context,
            "optimization_result": result.get("optimization_result"),
            "suggested_funds": self.context.get("suggested_funds", {}),
            "should_reset": result.get("should_reset", False)  # Pass through reset flag
        }
    
    def _determine_current_agent(self):
        """Determine current agent based on context state to prevent loops."""
        # Check in order of workflow - be very strict about not going backwards
        if not self.context.get("currency"):
            self.current_agent = "currency"
        elif self.context.get("currency") == "USD" and (not self.context.get("geography_constraints") or len(self.context.get("geography_constraints", {})) == 0):
            self.current_agent = "geography"
        elif not self.context.get("primary_risk_bucket"):
            self.current_agent = "risk_assessment"
        elif not self.context.get("suggested_funds") or len(self.context.get("suggested_funds", {})) == 0:
            self.current_agent = "fund_selection"
        elif not self.context.get("volatility_target_pct") and not self.context.get("drawdown_target_pct"):
            self.current_agent = "sub_risk_refinement"
        else:
            self.current_agent = "optimization"
        
        # CRITICAL: If we've moved past an agent, never go back
        # Check flags to prevent regression
        if self.context.get("_moved_to_geography") and self.current_agent == "currency":
            self.current_agent = "geography"
        if self.context.get("_moved_to_risk_assessment") and self.current_agent in ["currency", "geography"]:
            self.current_agent = "risk_assessment"
        if self.context.get("_moved_to_fund_selection") and self.current_agent in ["currency", "geography", "risk_assessment"]:
            self.current_agent = "fund_selection"
    
    def reset(self):
        """Reset coordinator state."""
        if self.db:
            try:
                self.db.clear_session(self.session_id)
            except Exception as e:
                print(f"Error clearing session: {e}")
        
        import uuid
        self.session_id = str(uuid.uuid4())
        self.current_agent = "currency"
        self.context = {
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
            "suggested_funds": {},
            "conversation_history": []
        }

