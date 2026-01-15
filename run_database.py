"""
Script to initialize and verify the database.
"""
from database import Database
import sqlite3

def main():
    print("=" * 60)
    print("Database Initialization and Verification")
    print("=" * 60)
    
    # Initialize database
    print("\n1. Initializing database...")
    db = Database()
    print(f"   [OK] Database initialized: {db.db_path}")
    
    # Verify tables exist
    print("\n2. Verifying database tables...")
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"   [OK] Found {len(tables)} tables:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"      - {table_name}: {count} records")
    
    # Show table schemas
    print("\n3. Table schemas:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"\n   {table_name}:")
        for col in columns:
            col_name, col_type = col[1], col[2]
            print(f"      - {col_name}: {col_type}")
    
    conn.close()
    
    # Test database operations
    print("\n4. Testing database operations...")
    test_session_id = "test_session_123"
    
    # Test save preferences
    test_preferences = {
        "currency": "USD",
        "primary_risk_bucket": "MEDIUM",
        "sub_risk_bucket": "MEDIUM_MEDIUM",
        "volatility_target_pct": 25.0,
        "fund_counts": {"debt": 1, "large_cap": 1, "mid_cap": 1},
        "asset_split_targets": {"debt": 25, "equity": 55, "balanced": 20}
    }
    db.save_user_preferences(test_session_id, test_preferences)
    print("   [OK] Saved test user preferences")
    
    # Test get preferences
    retrieved = db.get_user_preferences(test_session_id)
    if retrieved:
        print(f"   [OK] Retrieved preferences: Currency={retrieved['currency']}, Risk={retrieved['primary_risk_bucket']}")
    
    # Test save conversation
    db.save_conversation(test_session_id, "Hello", "Hi there!", {})
    print("   [OK] Saved test conversation")
    
    # Test get conversation history
    history = db.get_conversation_history(test_session_id)
    print(f"   [OK] Retrieved {len(history)} conversation(s)")
    
    print("\n" + "=" * 60)
    print("Database is ready and working correctly!")
    print("=" * 60)
    print(f"\nDatabase file: {db.db_path}")
    print("You can now use the database in your application.")

if __name__ == "__main__":
    main()

