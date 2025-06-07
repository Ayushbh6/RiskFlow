#!/usr/bin/env python3
"""
RiskFlow Dashboard Launcher

Script to start the Streamlit dashboard for RiskFlow MLOps system.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the RiskFlow Streamlit dashboard."""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dashboard_dir = project_root / "dashboard"
    dashboard_app = dashboard_dir / "app.py"
    
    # Ensure dashboard app exists
    if not dashboard_app.exists():
        print(f"❌ Dashboard app not found at {dashboard_app}")
        return 1
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(project_root)
    
    print("🚀 Starting RiskFlow MLOps Dashboard...")
    print(f"📁 Project root: {project_root}")
    print(f"🖥️  Dashboard: {dashboard_app}")
    print()
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("🔄 To stop the dashboard, press Ctrl+C")
    print()
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_app),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, cwd=project_root, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 