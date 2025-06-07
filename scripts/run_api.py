#!/usr/bin/env python3
"""
RiskFlow API Server Startup Script

Starts the FastAPI server for the Credit Risk MLOps Pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to Python path and change working directory
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to src directory so relative imports work
os.chdir(str(src_path))

def main():
    """Start the FastAPI server."""
    
    try:
        import uvicorn
        
        print("🚀 Starting RiskFlow Credit Risk API Server...")
        print("📍 API will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
        print("🔍 Health check: http://localhost:8000/health")
        print("\n💡 Press Ctrl+C to stop the server\n")
        
        # Start the server
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Auto-reload on code changes
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure you're in the project root and dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()