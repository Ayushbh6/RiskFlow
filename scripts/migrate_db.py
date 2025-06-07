#!/usr/bin/env python3
"""
Database Migration Script for RiskFlow

Handles schema updates for existing databases, including adding missing columns.
"""

import sys
from pathlib import Path
from sqlalchemy import inspect, text

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.database import engine, Base
from config.logging_config import get_logger

logger = get_logger(__name__)


def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def add_stage_column_to_model_registry():
    """Add the 'stage' column to model_registry table if it doesn't exist."""
    table_name = 'model_registry'
    column_name = 'stage'
    
    if not check_table_exists(table_name):
        logger.info(f"Table '{table_name}' does not exist. Creating all tables...")
        Base.metadata.create_all(bind=engine)
        return
    
    if check_column_exists(table_name, column_name):
        logger.info(f"Column '{column_name}' already exists in '{table_name}'")
        return
    
    logger.info(f"Adding '{column_name}' column to '{table_name}' table...")
    
    with engine.begin() as conn:
        # Add the stage column with a default value
        conn.execute(text(f"""
            ALTER TABLE {table_name} 
            ADD COLUMN {column_name} VARCHAR DEFAULT 'development'
        """))
        
        # Create index on stage column
        conn.execute(text(f"""
            CREATE INDEX ix_{table_name}_{column_name} 
            ON {table_name} ({column_name})
        """))
    
    logger.info(f"Successfully added '{column_name}' column to '{table_name}'")


def run_migrations():
    """Run all database migrations."""
    print("🔄 Running database migrations...")
    
    try:
        # Add stage column to model_registry
        add_stage_column_to_model_registry()
        
        # Ensure all tables exist
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database migrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"❌ Migration failed: {e}")
        raise


def main():
    """Main migration entry point."""
    try:
        run_migrations()
        print("\n🎉 Database is now up to date!")
        print("\n🚀 You can now start the application:")
        print("   ./start-app.sh")
        
    except Exception as e:
        print(f"\n❌ Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()