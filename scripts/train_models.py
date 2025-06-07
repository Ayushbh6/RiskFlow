#!/usr/bin/env python3
"""
RiskFlow Model Training Script

Trains credit risk models and registers them in MLflow.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def main():
    """Train and register credit risk models."""
    
    try:
        from models.model_training import ModelTrainingPipeline
        from data.ingestion import DataIngestionPipeline
        from utils.database import db_manager
        from config.settings import get_settings
        
        print("🤖 Starting RiskFlow Model Training...")
        print("=" * 50)
        
        # Initialize components
        settings = get_settings()
        trainer = ModelTrainingPipeline()
        
        # Check if we have real data
        print("📊 Checking for existing training data...")
        existing_data = db_manager.get_credit_data(limit=10)
        
        if not existing_data:
            print("⚠️  No real data found in database")
            print("❌ CANNOT TRAIN MODEL WITHOUT REAL DATA")
            print("\n🔧 Please run the following command first to fetch real loan data:")
            print("   python scripts/fetch_real_loan_data.py")
            print("\nThis will fetch REAL historical loan performance data from public sources.")
            sys.exit(1)
        
        # Run training experiment
        print("\n🚀 Starting model training experiment...")
        print("=" * 50)
        
        experiment_results = trainer.run_training_experiment(
            experiment_name="initial_model_training",
            data_limit=None,  # Use all available data
            pd_model_types=["ensemble", "random_forest"],
            lgd_model_types=["gradient_boosting"]
        )
        
        # Display results
        if experiment_results and 'best_model' in experiment_results:
            best = experiment_results['best_model']
            print("\n🏆 Best Model Results:")
            print("=" * 50)
            print(f"Model Type: PD={best['pd_type']}, LGD={best['lgd_type']}")
            print(f"Combined Score: {best['combined_score']:.3f}")
            
            if 'pd_metrics' in best['metrics']:
                pd_metrics = best['metrics']['pd_metrics']
                print(f"\nPD Model Performance:")
                print(f"  - AUC: {pd_metrics.get('cv_auc_mean', 0):.3f}")
                print(f"  - Precision: {pd_metrics.get('cv_precision_mean', 0):.3f}")
                print(f"  - Recall: {pd_metrics.get('cv_recall_mean', 0):.3f}")
            
            if 'lgd_metrics' in best['metrics']:
                lgd_metrics = best['metrics']['lgd_metrics']
                print(f"\nLGD Model Performance:")
                print(f"  - RMSE: {lgd_metrics.get('cv_rmse_mean', 0):.3f}")
                print(f"  - R²: {lgd_metrics.get('cv_r2_mean', 0):.3f}")
            
            if 'registry_id' in best:
                print(f"\n✅ Model registered with ID: {best['registry_id']}")
        
        print("\n🎉 Model training completed successfully!")
        print("=" * 50)
        print("📊 Models are now available for predictions via the API")
        print("🔗 API endpoint: http://localhost:8000/predict")
        print("📈 MLflow tracking: http://localhost:5000 (if MLflow UI is running)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure you're in the project root and dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        print("🔍 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()