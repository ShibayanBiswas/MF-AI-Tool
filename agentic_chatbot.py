"""
Agentic AI Chatbot with hierarchical agentic architecture.
Uses specialized sub-agents for different tasks with proper tool calling.
"""
from teams import CoordinatorAgent


class AgenticPortfolioAgent:
    """
    Agentic AI agent that autonomously manages portfolio recommendation conversations.
    Uses hierarchical coordinator agent system with specialized sub-agents.
    """
    
    def __init__(self, session_id="default"):
        # Use hierarchical coordinator agent
        self.coordinator = CoordinatorAgent(session_id)
        self.session_id = session_id
        # Keep state for backward compatibility
        self.state = self.coordinator.context
        # Keep db reference for backward compatibility
        self.db = self.coordinator.db
    
    def chat(self, user_message: str) -> dict:
        """Process user message through hierarchical agent system."""
        # Use coordinator agent
        result = self.coordinator.chat(user_message)
        # Sync state for backward compatibility
        self._sync_state()
        return result
    
    def _sync_state(self):
        """Sync state from coordinator context."""
        self.state = self.coordinator.context.copy()
        self.session_id = self.coordinator.session_id
    
    def reset(self):
        """Reset agent state and clear database cache for this session."""
        self.coordinator.reset()
        self._sync_state()
        print(f"New session started: {self.session_id}")
    
    def _build_optimization_payload(self) -> dict:
        """Build optimization payload from current state (for backward compatibility)."""
        return {
            "currency": self.state.get("currency"),
            "primary_risk_bucket": self.state.get("primary_risk_bucket"),
            "sub_risk_bucket": self.state.get("sub_risk_bucket"),
            "volatility_target_pct": self.state.get("volatility_target_pct"),
            "drawdown_target_pct": self.state.get("drawdown_target_pct"),
            "fund_counts": self.state.get("fund_counts", {}),
            "asset_split_targets": self.state.get("asset_split_targets", {}),
            "geography_constraints": self.state.get("geography_constraints", {}),
            "tax_saver_target_pct": self.state.get("tax_saver_target_pct", 0)
        }


# Backward compatibility
PortfolioChatbot = AgenticPortfolioAgent
