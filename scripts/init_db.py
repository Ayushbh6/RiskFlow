#!/usr/bin/env python3
"""
Initialize RiskFlow Database with Real Data Schema

Creates database tables for storing real metrics - NO FAKE DATA
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from sqlalchemy import create_engine
from utils.database import Base, engine
from config.logging_config import get_logger

logger = get_logger(__name__)

def create_database():
    """Create database with real data tables using SQLAlchemy models."""
    
    print("🗄️ Creating RiskFlow database tables...")
    
    try:
        # Create all tables defined in SQLAlchemy models
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Check database path for SQLite
        if "sqlite" in str(engine.url):
            db_path = str(engine.url).replace("sqlite:///", "")
            print(f"🗄️ Database ready at: {db_path}")
        
        print("💡 Database is empty - ready for real data")
        print("📊 When you make predictions via API, they will appear here")
        
    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        raise

def main():
    """Initialize the database."""
    try:
        create_database()
        print("\n✅ RiskFlow database initialized successfully!")
        print("\n🚀 Next steps:")
        print("1. Start API server: python scripts/run_api.py")
        print("2. Make predictions to populate with real data")
        print("3. View dashboard: streamlit run dashboard/app.py")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()