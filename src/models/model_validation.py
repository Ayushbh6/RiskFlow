"""
Model validation and testing framework for RiskFlow Credit Risk MLOps.
Ensures model quality and performance standards are met.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

from models.credit_risk_model import CreditRiskModel
from config.settings import get_settings
from config.logging_config import get_logger
from utils.database import DatabaseManager, get_db_session
from utils.exceptions import ModelValidationError
from data.preprocessing import CreditRiskFeatureEngineer
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()


class ModelValidator:
    """
    Comprehensive model validation framework.
    Tests model performance, stability, and business constraints.
    """
    
    def __init__(self, model: CreditRiskModel):
        """
        Initialize validator with a trained model.
        
        Args:
            model: Trained CreditRiskModel instance
        """
        self.model = model
        self.validation_results = {}
        
    def validate_model(
        self,
        X_test: pd.DataFrame,
        y_test_default: pd.Series,
        y_test_lgd: pd.Series,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run complete model validation suite.
        
        Args:
            X_test: Test features
            y_test_default: Test default labels
            y_test_lgd: Test LGD values
            thresholds: Performance thresholds for pass/fail
        
        Returns:
            Validation results dictionary
        """
        try:
            logger.info(f"Running model validation on {len(X_test)} test samples")
            
            # Default thresholds
            if thresholds is None:
                thresholds = {
                    'pd_auc_min': 0.75,
                    'lgd_rmse_max': 0.25,
                    'calibration_error_max': 0.05,
                    'business_rules_pass_rate': 0.95
                }
            
            # Get predictions
            predictions = self.model.predict(X_test)
            
            # Validate PD model
            pd_validation = self._validate_pd_model(
                y_test_default,
                predictions['probability_of_default'],
                threshold=thresholds['pd_auc_min']
            )
            
            # Validate LGD model
            lgd_validation = self._validate_lgd_model(
                y_test_lgd,
                predictions['loss_given_default'],
                y_test_default,
                threshold=thresholds['lgd_rmse_max']
            )
            
            # Validate calibration
            calibration_validation = self._validate_calibration(
                y_test_default,
                predictions['probability_of_default'],
                threshold=thresholds['calibration_error_max']
            )
            
            # Validate business rules
            business_validation = self._validate_business_rules(
                X_test,
                predictions,
                threshold=thresholds['business_rules_pass_rate']
            )
            
            # Stability validation
            stability_validation = self._validate_stability(
                X_test,
                predictions
            )
            
            # Compile results
            self.validation_results = {
                'pd_validation': pd_validation,
                'lgd_validation': lgd_validation,
                'calibration_validation': calibration_validation,
                'business_validation': business_validation,
                'stability_validation': stability_validation,
                'overall_pass': all([
                    pd_validation['pass'],
                    lgd_validation['pass'],
                    calibration_validation['pass'],
                    business_validation['pass']
                ]),
                'validation_timestamp': get_utc_now().isoformat(),
                'test_samples': len(X_test),
                'thresholds_used': thresholds
            }
            
            logger.info(f"Model validation completed. Overall pass: {self.validation_results['overall_pass']}")
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise
    
    def _validate_pd_model(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Validate PD model performance."""
        try:
            # Calculate metrics
            auc = roc_auc_score(y_true, y_pred_proba)
            
            # Get optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Binary predictions at optimal threshold
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Gini coefficient
            gini = 2 * auc - 1
            
            validation = {
                'auc': float(auc),
                'gini': float(gini),
                'optimal_threshold': float(optimal_threshold),
                'precision': float(report['1']['precision']),
                'recall': float(report['1']['recall']),
                'f1_score': float(report['1']['f1-score']),
                'confusion_matrix': cm.tolist(),
                'pass': auc >= threshold,
                'message': f"PD model AUC: {auc:.4f} ({'PASS' if auc >= threshold else 'FAIL'})"
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"PD validation failed: {str(e)}")
            return {'pass': False, 'message': f"PD validation error: {str(e)}"}
    
    def _validate_lgd_model(
        self,
        y_true_lgd: pd.Series,
        y_pred_lgd: np.ndarray,
        y_true_default: pd.Series,
        threshold: float
    ) -> Dict[str, Any]:
        """Validate LGD model performance."""
        try:
            # Only validate on defaulted loans
            default_mask = y_true_default == 1
            
            if default_mask.sum() == 0:
                return {
                    'pass': True,
                    'message': "No defaulted loans in test set for LGD validation"
                }
            
            y_true_lgd_defaulted = y_true_lgd[default_mask]
            y_pred_lgd_defaulted = y_pred_lgd[default_mask]
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true_lgd_defaulted, y_pred_lgd_defaulted))
            mae = mean_absolute_error(y_true_lgd_defaulted, y_pred_lgd_defaulted)
            r2 = r2_score(y_true_lgd_defaulted, y_pred_lgd_defaulted)
            
            # Check if predictions are within valid range
            out_of_range = ((y_pred_lgd_defaulted < 0) | (y_pred_lgd_defaulted > 1)).sum()
            
            validation = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'out_of_range_predictions': int(out_of_range),
                'defaulted_samples': int(default_mask.sum()),
                'pass': rmse <= threshold and out_of_range == 0,
                'message': f"LGD model RMSE: {rmse:.4f} ({'PASS' if rmse <= threshold else 'FAIL'})"
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"LGD validation failed: {str(e)}")
            return {'pass': False, 'message': f"LGD validation error: {str(e)}"}
    
    def _validate_calibration(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        threshold: float,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Validate model calibration."""
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins, strategy='quantile'
            )
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Hosmer-Lemeshow test approximation
            bin_edges = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
            bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
            
            calibration_table = []
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() > 0:
                    observed = y_true[mask].mean()
                    expected = y_pred_proba[mask].mean()
                    count = mask.sum()
                    calibration_table.append({
                        'bin': i + 1,
                        'count': int(count),
                        'observed_rate': float(observed),
                        'expected_rate': float(expected),
                        'difference': float(observed - expected)
                    })
            
            validation = {
                'calibration_error': float(calibration_error),
                'calibration_table': calibration_table,
                'pass': calibration_error <= threshold,
                'message': f"Calibration error: {calibration_error:.4f} ({'PASS' if calibration_error <= threshold else 'FAIL'})"
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"Calibration validation failed: {str(e)}")
            return {'pass': False, 'message': f"Calibration validation error: {str(e)}"}
    
    def _validate_business_rules(
        self,
        X_test: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        threshold: float
    ) -> Dict[str, Any]:
        """Validate business rules and constraints."""
        try:
            violations = []
            total_checks = 0
            
            # Rule 1: Higher credit scores should have lower PD
            if 'credit_score_normalized' in X_test.columns:
                total_checks += 1
                high_score_mask = X_test['credit_score_normalized'] > 0.8
                if high_score_mask.sum() > 0:
                    avg_pd_high_score = predictions['probability_of_default'][high_score_mask].mean()
                    if avg_pd_high_score > 0.1:  # PD should be <10% for high credit scores
                        violations.append(f"High credit score avg PD too high: {avg_pd_high_score:.2%}")
            
            # Rule 2: Risk ratings should be monotonic with PD
            total_checks += 1
            risk_ratings = predictions['risk_rating']
            pd_by_rating = pd.DataFrame({
                'rating': risk_ratings,
                'pd': predictions['probability_of_default']
            }).groupby('rating')['pd'].mean()
            
            if not pd_by_rating.is_monotonic_increasing:
                violations.append("Risk ratings not monotonic with PD")
            
            # Rule 3: Expected loss should be reasonable
            total_checks += 1
            el = predictions['expected_loss']
            if el.max() > 1.0 or el.min() < 0:
                violations.append(f"Expected loss out of range: [{el.min():.3f}, {el.max():.3f}]")
            
            # Rule 4: DTI ratio impact
            if 'debt_to_income' in X_test.columns:
                total_checks += 1
                high_dti_mask = X_test['debt_to_income'] > 0.5
                if high_dti_mask.sum() > 0:
                    avg_pd_high_dti = predictions['probability_of_default'][high_dti_mask].mean()
                    avg_pd_low_dti = predictions['probability_of_default'][~high_dti_mask].mean()
                    if avg_pd_high_dti <= avg_pd_low_dti:
                        violations.append("High DTI not showing higher risk")
            
            pass_rate = (total_checks - len(violations)) / total_checks if total_checks > 0 else 1.0
            
            validation = {
                'total_rules_checked': total_checks,
                'violations': violations,
                'violation_count': len(violations),
                'pass_rate': float(pass_rate),
                'pass': pass_rate >= threshold,
                'message': f"Business rules pass rate: {pass_rate:.1%} ({'PASS' if pass_rate >= threshold else 'FAIL'})"
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"Business rules validation failed: {str(e)}")
            return {'pass': False, 'message': f"Business rules validation error: {str(e)}"}
    
    def _validate_stability(
        self,
        X_test: pd.DataFrame,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Validate model stability and robustness."""
        try:
            stability_checks = {}
            
            # Check prediction distribution
            pd_preds = predictions['probability_of_default']
            stability_checks['pd_distribution'] = {
                'mean': float(pd_preds.mean()),
                'std': float(pd_preds.std()),
                'min': float(pd_preds.min()),
                'max': float(pd_preds.max()),
                'zeros': int((pd_preds == 0).sum()),
                'ones': int((pd_preds == 1).sum())
            }
            
            # Check for numerical stability
            stability_checks['numerical_issues'] = {
                'inf_values': int(np.isinf(pd_preds).sum()),
                'nan_values': int(np.isnan(pd_preds).sum())
            }
            
            # Feature sensitivity check (small perturbations)
            if len(X_test) > 100:
                sample_idx = np.random.choice(len(X_test), 100, replace=False)
                X_sample = X_test.iloc[sample_idx].copy()
                
                # Add small noise to numeric features
                numeric_cols = X_sample.select_dtypes(include=[np.number]).columns
                X_perturbed = X_sample.copy()
                X_perturbed[numeric_cols] *= (1 + np.random.normal(0, 0.01, size=(100, len(numeric_cols))))
                
                # Get predictions on perturbed data
                perturbed_preds = self.model.predict(X_perturbed)
                
                # Calculate stability metric
                pd_diff = np.abs(predictions['probability_of_default'][sample_idx] - 
                               perturbed_preds['probability_of_default'])
                
                stability_checks['perturbation_sensitivity'] = {
                    'mean_absolute_change': float(pd_diff.mean()),
                    'max_absolute_change': float(pd_diff.max()),
                    'stable': bool(pd_diff.max() < 0.1)  # <10% change is stable
                }
            
            return {
                'stability_checks': stability_checks,
                'is_stable': all([
                    stability_checks['numerical_issues']['inf_values'] == 0,
                    stability_checks['numerical_issues']['nan_values'] == 0,
                    stability_checks.get('perturbation_sensitivity', {}).get('stable', True)
                ])
            }
            
        except Exception as e:
            logger.error(f"Stability validation failed: {str(e)}")
            return {'is_stable': False, 'message': f"Stability validation error: {str(e)}"}
    
    def generate_validation_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            output_path: Path to save report (optional)
        
        Returns:
            Report as string
        """
        if not self.validation_results:
            return "No validation results available. Run validate_model() first."
        
        report = []
        report.append("=" * 60)
        report.append("CREDIT RISK MODEL VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Date: {self.validation_results.get('validation_timestamp', 'N/A')}")
        report.append(f"Test Samples: {self.validation_results.get('test_samples', 'N/A')}")
        report.append(f"Overall Result: {'PASS' if self.validation_results.get('overall_pass') else 'FAIL'}")
        report.append("")
        
        # PD Model Results
        report.append("PD MODEL VALIDATION")
        report.append("-" * 40)
        pd_val = self.validation_results.get('pd_validation', {})
        report.append(f"AUC Score: {pd_val.get('auc', 0):.4f}")
        report.append(f"Gini Coefficient: {pd_val.get('gini', 0):.4f}")
        report.append(f"Result: {pd_val.get('message', 'N/A')}")
        report.append("")
        
        # LGD Model Results
        report.append("LGD MODEL VALIDATION")
        report.append("-" * 40)
        lgd_val = self.validation_results.get('lgd_validation', {})
        report.append(f"RMSE: {lgd_val.get('rmse', 0):.4f}")
        report.append(f"MAE: {lgd_val.get('mae', 0):.4f}")
        report.append(f"R²: {lgd_val.get('r2', 0):.4f}")
        report.append(f"Result: {lgd_val.get('message', 'N/A')}")
        report.append("")
        
        # Calibration Results
        report.append("CALIBRATION VALIDATION")
        report.append("-" * 40)
        cal_val = self.validation_results.get('calibration_validation', {})
        report.append(f"Calibration Error: {cal_val.get('calibration_error', 0):.4f}")
        report.append(f"Result: {cal_val.get('message', 'N/A')}")
        report.append("")
        
        # Business Rules Results
        report.append("BUSINESS RULES VALIDATION")
        report.append("-" * 40)
        bus_val = self.validation_results.get('business_validation', {})
        report.append(f"Rules Checked: {bus_val.get('total_rules_checked', 0)}")
        report.append(f"Violations: {bus_val.get('violation_count', 0)}")
        report.append(f"Pass Rate: {bus_val.get('pass_rate', 0):.1%}")
        report.append(f"Result: {bus_val.get('message', 'N/A')}")
        
        if bus_val.get('violations'):
            report.append("\nViolations:")
            for violation in bus_val['violations']:
                report.append(f"  - {violation}")
        report.append("")
        
        # Stability Results
        report.append("STABILITY VALIDATION")
        report.append("-" * 40)
        stab_val = self.validation_results.get('stability_validation', {})
        report.append(f"Model Stable: {'Yes' if stab_val.get('is_stable') else 'No'}")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text


def validate_model_for_production(
    model: CreditRiskModel,
    test_data: Tuple[pd.DataFrame, pd.Series, pd.Series]
) -> bool:
    """
    Validate if model is ready for production deployment.
    
    Args:
        model: Trained credit risk model
        test_data: Tuple of (X_test, y_test_default, y_test_lgd)
    
    Returns:
        True if model passes all validation checks
    """
    validator = ModelValidator(model)
    X_test, y_test_default, y_test_lgd = test_data
    
    results = validator.validate_model(X_test, y_test_default, y_test_lgd)
    
    if results['overall_pass']:
        logger.info("Model passed all validation checks and is ready for production")
    else:
        logger.warning("Model failed validation checks")
    
    return results['overall_pass']