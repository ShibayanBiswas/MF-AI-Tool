"""
Database storage for conversations, user preferences, and portfolio data.
Uses SQLite for simplicity.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional

class Database:
    def __init__(self, db_path="portfolio_chatbot.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                state_json TEXT
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                currency TEXT,
                primary_risk_bucket TEXT,
                sub_risk_bucket TEXT,
                volatility_target_pct REAL,
                drawdown_target_pct REAL,
                fund_counts_json TEXT,
                asset_split_targets_json TEXT,
                geography_constraints_json TEXT,
                tax_saver_target_pct REAL,
                suggested_funds_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add suggested_funds_json column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE user_preferences ADD COLUMN suggested_funds_json TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Portfolio results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                optimization_payload_json TEXT,
                weights_json TEXT,
                funds_json TEXT,
                model_used TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, user_message: str, bot_response: str, state: Dict):
        """Save a conversation message."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, user_message, bot_response, state_json)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_message, bot_response, json.dumps(state, default=str)))
        conn.commit()
        conn.close()
    
    def save_user_preferences(self, session_id: str, preferences: Dict):
        """Save or update user preferences."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT id FROM user_preferences WHERE session_id = ?", (session_id,))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute("""
                UPDATE user_preferences SET
                    currency = ?,
                    primary_risk_bucket = ?,
                    sub_risk_bucket = ?,
                    volatility_target_pct = ?,
                    drawdown_target_pct = ?,
                    fund_counts_json = ?,
                    asset_split_targets_json = ?,
                    geography_constraints_json = ?,
                    tax_saver_target_pct = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (
                preferences.get("currency"),
                preferences.get("primary_risk_bucket"),
                preferences.get("sub_risk_bucket"),
                preferences.get("volatility_target_pct"),
                preferences.get("drawdown_target_pct"),
                json.dumps(preferences.get("fund_counts", {})),
                json.dumps(preferences.get("asset_split_targets", {})),
                json.dumps(preferences.get("geography_constraints", {})),
                preferences.get("tax_saver_target_pct"),
                session_id
            ))
        else:
            cursor.execute("""
                INSERT INTO user_preferences (
                    session_id, currency, primary_risk_bucket, sub_risk_bucket,
                    volatility_target_pct, drawdown_target_pct, fund_counts_json,
                    asset_split_targets_json, geography_constraints_json, tax_saver_target_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                preferences.get("currency"),
                preferences.get("primary_risk_bucket"),
                preferences.get("sub_risk_bucket"),
                preferences.get("volatility_target_pct"),
                preferences.get("drawdown_target_pct"),
                json.dumps(preferences.get("fund_counts", {})),
                json.dumps(preferences.get("asset_split_targets", {})),
                json.dumps(preferences.get("geography_constraints", {})),
                preferences.get("tax_saver_target_pct")
            ))
        
        conn.commit()
        conn.close()
    
    def save_portfolio_result(self, session_id: str, payload: Dict, result: Dict):
        """Save portfolio optimization result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio_results (
                session_id, optimization_payload_json, weights_json, funds_json, model_used
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            json.dumps(payload, default=str),
            json.dumps(result.get("weights", {}), default=str),
            json.dumps(result.get("funds", []), default=str),
            result.get("model_used", "unknown")
        ))
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, session_id: str) -> Optional[Dict]:
        """Get user preferences for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_preferences WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "session_id": row[1],
                "currency": row[2],
                "primary_risk_bucket": row[3],
                "sub_risk_bucket": row[4],
                "volatility_target_pct": row[5],
                "drawdown_target_pct": row[6],
                "fund_counts": json.loads(row[7]) if row[7] else {},
                "asset_split_targets": json.loads(row[8]) if row[8] else {},
                "geography_constraints": json.loads(row[9]) if row[9] else {},
                "tax_saver_target_pct": row[10]
            }
        return None
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, bot_response, timestamp, state_json
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "user_message": row[0],
                "bot_response": row[1],
                "timestamp": row[2],
                "state": json.loads(row[3]) if row[3] else {}
            }
            for row in reversed(rows)  # Reverse to get chronological order
        ]
    
    def clear_session(self, session_id: str):
        """Clear all data for a session (reset)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete conversations
        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        
        # Delete user preferences
        cursor.execute("DELETE FROM user_preferences WHERE session_id = ?", (session_id,))
        
        # Delete portfolio results
        cursor.execute("DELETE FROM portfolio_results WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def save_suggested_funds(self, session_id: str, suggested_funds: Dict):
        """Save suggested funds for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update user_preferences with suggested_funds
        cursor.execute("""
            UPDATE user_preferences 
            SET suggested_funds_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (json.dumps(suggested_funds, default=str), session_id))
        
        conn.commit()
        conn.close()
    
    def get_suggested_funds(self, session_id: str) -> Optional[Dict]:
        """Get suggested funds for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add suggested_funds_json column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE user_preferences ADD COLUMN suggested_funds_json TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        cursor.execute("SELECT suggested_funds_json FROM user_preferences WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return json.loads(row[0])
        return None

