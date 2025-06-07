"""
Prompt templates for LLM-powered credit risk analysis and documentation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class PromptTemplates:
    """Collection of prompt templates for credit risk analysis."""
    
    @staticmethod
    def risk_analysis_prompt(
        borrower_data: Dict[str, Any],
        model_prediction: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate prompt for comprehensive credit risk analysis.
        
        Args:
            borrower_data: Borrower information and financial metrics
            model_prediction: Model prediction results (PD, LGD, etc.)
            market_context: Current market conditions and economic indicators
        
        Returns:
            List of messages for LLM chat completion
        """
        system_prompt = """You are a senior credit risk analyst with 15+ years of experience in financial risk assessment. Your role is to provide comprehensive, actionable credit risk analysis based on quantitative models and qualitative factors.

Key responsibilities:
1. Interpret model predictions in business context
2. Identify key risk factors and mitigants
3. Provide clear, actionable recommendations
4. Explain complex risk concepts in accessible language
5. Consider both quantitative metrics and qualitative factors

Analysis framework:
- Credit Quality Assessment
- Risk Factor Analysis
- Market Context Integration
- Recommendation Synthesis
- Monitoring Suggestions"""

                # Format borrower data
        def format_currency(value):
            return f"${value:,}" if isinstance(value, (int, float)) else str(value)
        
        def format_percentage(value):
            return f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
        
        def format_ratio(value):
            return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
        
        borrower_summary = f"""
BORROWER PROFILE:
- Borrower ID: {borrower_data.get('borrower_id', 'N/A')}
- Industry: {borrower_data.get('industry', 'N/A')}
- Company Size: {borrower_data.get('company_size', 'N/A')}
- Years in Business: {borrower_data.get('years_in_business', 'N/A')}
- Annual Revenue: {format_currency(borrower_data.get('annual_revenue', 'N/A'))}
- Total Assets: {format_currency(borrower_data.get('total_assets', 'N/A'))}
- Total Debt: {format_currency(borrower_data.get('total_debt', 'N/A'))}
- Credit Score: {borrower_data.get('credit_score', 'N/A')}
- Debt-to-Income Ratio: {format_percentage(borrower_data.get('debt_to_income_ratio', 'N/A'))}
- Current Ratio: {format_ratio(borrower_data.get('current_ratio', 'N/A'))}
- ROE: {format_percentage(borrower_data.get('roe', 'N/A'))}
"""

        # Format model predictions
        prediction_summary = f"""
MODEL PREDICTIONS:
- Probability of Default (PD): {format_percentage(model_prediction.get('pd', 'N/A'))}
- Loss Given Default (LGD): {format_percentage(model_prediction.get('lgd', 'N/A'))}
- Expected Loss: {format_percentage(model_prediction.get('expected_loss', 'N/A'))}
- Risk Rating: {model_prediction.get('risk_rating', 'N/A')}
- Model Confidence: {format_percentage(model_prediction.get('confidence', 'N/A'))}
- Key Risk Factors: {', '.join(model_prediction.get('top_risk_factors', []))}
"""

        # Format market context if available
        market_summary = ""
        if market_context:
            market_summary = f"""
MARKET CONTEXT:
- Economic Environment: {market_context.get('economic_environment', 'N/A')}
- Interest Rate Environment: {market_context.get('interest_rates', 'N/A')}
- Industry Outlook: {market_context.get('industry_outlook', 'N/A')}
- Credit Market Conditions: {market_context.get('credit_conditions', 'N/A')}
"""

        user_prompt = f"""Please provide a comprehensive credit risk analysis for this borrower:

{borrower_summary}

{prediction_summary}

{market_summary}

Please structure your analysis as follows:

## EXECUTIVE SUMMARY
- Overall risk assessment (1-2 sentences)
- Key recommendation

## CREDIT QUALITY ASSESSMENT
- Strengths and positive factors
- Weaknesses and concerns
- Overall creditworthiness evaluation

## RISK FACTOR ANALYSIS
- Primary risk drivers
- Secondary risk considerations
- Risk mitigants and protective factors

## MODEL INTERPRETATION
- Analysis of model predictions
- Confidence assessment
- Key variables driving the prediction

## MARKET CONTEXT IMPACT
- How current market conditions affect this credit
- Industry-specific considerations
- Economic environment implications

## RECOMMENDATIONS
- Credit decision recommendation (Approve/Decline/Conditional)
- Suggested terms and conditions
- Risk monitoring requirements
- Potential risk mitigation strategies

## NEXT STEPS
- Required additional information
- Ongoing monitoring recommendations
- Review timeline suggestions

Please provide specific, actionable insights that would help a credit committee make an informed decision."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def model_documentation_prompt(
        model_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        feature_importance: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Generate prompt for model documentation.
        
        Args:
            model_info: Model metadata and configuration
            performance_metrics: Model performance statistics
            feature_importance: Feature importance rankings
        
        Returns:
            List of messages for LLM chat completion
        """
        system_prompt = """You are a machine learning engineer and model risk management specialist. Your role is to create comprehensive, regulatory-compliant model documentation that explains complex ML models in clear, accessible language for both technical and non-technical stakeholders.

Documentation standards:
- Clear, jargon-free explanations
- Regulatory compliance (SR 11-7, CECL, Basel III)
- Technical accuracy with business context
- Risk management focus
- Stakeholder-appropriate detail levels"""

        def format_number(value):
            return f"{value:,}" if isinstance(value, (int, float)) else str(value)
        
        model_summary = f"""
MODEL INFORMATION:
- Model ID: {model_info.get('model_id', 'N/A')}
- Model Type: {model_info.get('model_type', 'N/A')}
- Algorithm: {model_info.get('algorithm', 'N/A')}
- Training Date: {model_info.get('training_date', 'N/A')}
- Model Version: {model_info.get('version', 'N/A')}
- Training Data Period: {model_info.get('data_period', 'N/A')}
- Number of Features: {model_info.get('num_features', 'N/A')}
- Training Samples: {format_number(model_info.get('training_samples', 'N/A'))}
"""

        def format_metric(value):
            return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
        
        performance_summary = f"""
PERFORMANCE METRICS:
- AUC-ROC: {format_metric(performance_metrics.get('auc_roc', 'N/A'))}
- Accuracy: {format_metric(performance_metrics.get('accuracy', 'N/A'))}
- Precision: {format_metric(performance_metrics.get('precision', 'N/A'))}
- Recall: {format_metric(performance_metrics.get('recall', 'N/A'))}
- F1-Score: {format_metric(performance_metrics.get('f1_score', 'N/A'))}
- Gini Coefficient: {format_metric(performance_metrics.get('gini', 'N/A'))}
- KS Statistic: {format_metric(performance_metrics.get('ks_statistic', 'N/A'))}
"""

        features_summary = "TOP FEATURES:\n"
        for i, feature in enumerate(feature_importance[:10], 1):
            features_summary += f"- {i}. {feature.get('name', 'N/A')}: {feature.get('importance', 0):.3f}\n"

        user_prompt = f"""Please create comprehensive model documentation for this credit risk model:

{model_summary}

{performance_summary}

{features_summary}

Please structure the documentation as follows:

## MODEL OVERVIEW
- Purpose and business objective
- Model type and methodology
- Target variable definition
- Use case and applications

## MODEL DEVELOPMENT
- Data sources and preparation
- Feature engineering approach
- Model selection rationale
- Training methodology
- Validation approach

## MODEL PERFORMANCE
- Performance metrics interpretation
- Validation results
- Backtesting outcomes
- Benchmark comparisons
- Stability analysis

## FEATURE ANALYSIS
- Key predictive variables
- Feature importance interpretation
- Business logic validation
- Correlation analysis
- Feature stability

## MODEL LIMITATIONS
- Known limitations and constraints
- Data quality considerations
- Model assumptions
- Potential biases
- Uncertainty quantification

## GOVERNANCE & CONTROLS
- Model validation framework
- Monitoring requirements
- Performance thresholds
- Escalation procedures
- Review schedule

## REGULATORY CONSIDERATIONS
- Regulatory compliance status
- Documentation requirements
- Audit trail maintenance
- Risk management integration

## IMPLEMENTATION GUIDANCE
- Deployment requirements
- Integration considerations
- User training needs
- Change management process

Please ensure the documentation is suitable for model risk management review and regulatory examination."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def risk_commentary_prompt(
        portfolio_metrics: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate prompt for portfolio risk commentary.
        
        Args:
            portfolio_metrics: Current portfolio risk metrics
            trend_analysis: Historical trend analysis
            market_conditions: Current market environment
        
        Returns:
            List of messages for LLM chat completion
        """
        system_prompt = """You are a chief risk officer with extensive experience in credit portfolio management. Your role is to provide executive-level risk commentary that synthesizes quantitative metrics with qualitative market insights for senior management and board reporting.

Commentary principles:
- Executive-level perspective
- Forward-looking risk assessment
- Actionable strategic insights
- Clear risk-return trade-offs
- Regulatory and stakeholder awareness"""

        portfolio_summary = f"""
PORTFOLIO METRICS:
- Total Exposure: ${portfolio_metrics.get('total_exposure', 0):,.0f}
- Number of Accounts: {portfolio_metrics.get('num_accounts', 0):,}
- Average PD: {portfolio_metrics.get('avg_pd', 0):.2%}
- Average LGD: {portfolio_metrics.get('avg_lgd', 0):.2%}
- Expected Loss: ${portfolio_metrics.get('expected_loss', 0):,.0f}
- Risk-Adjusted Return: {portfolio_metrics.get('risk_adjusted_return', 0):.2%}
- Concentration Risk: {portfolio_metrics.get('concentration_risk', 'N/A')}
"""

        trend_summary = f"""
TREND ANALYSIS:
- PD Trend (3M): {trend_analysis.get('pd_trend_3m', 'N/A')}
- Loss Rate Trend (12M): {trend_analysis.get('loss_trend_12m', 'N/A')}
- Portfolio Growth: {trend_analysis.get('portfolio_growth', 'N/A')}
- Credit Quality Migration: {trend_analysis.get('quality_migration', 'N/A')}
- Delinquency Trends: {trend_analysis.get('delinquency_trend', 'N/A')}
"""

        market_summary = f"""
MARKET CONDITIONS:
- Economic Outlook: {market_conditions.get('economic_outlook', 'N/A')}
- Interest Rate Environment: {market_conditions.get('interest_rates', 'N/A')}
- Credit Spreads: {market_conditions.get('credit_spreads', 'N/A')}
- Regulatory Environment: {market_conditions.get('regulatory_environment', 'N/A')}
- Competitive Landscape: {market_conditions.get('competitive_landscape', 'N/A')}
"""

        user_prompt = f"""Please provide executive-level risk commentary for our credit portfolio:

{portfolio_summary}

{trend_summary}

{market_summary}

Please structure your commentary as follows:

## EXECUTIVE SUMMARY
- Overall portfolio risk assessment
- Key risk themes and concerns
- Strategic recommendations

## CURRENT RISK PROFILE
- Portfolio composition analysis
- Risk concentration assessment
- Credit quality evaluation
- Performance vs. targets

## TREND ANALYSIS
- Historical performance trends
- Emerging risk patterns
- Early warning indicators
- Comparative analysis

## MARKET ENVIRONMENT IMPACT
- Economic environment implications
- Interest rate sensitivity
- Competitive pressures
- Regulatory considerations

## FORWARD-LOOKING ASSESSMENT
- Stress scenario analysis
- Potential risk scenarios
- Emerging threats and opportunities
- Strategic positioning

## RISK MANAGEMENT ACTIONS
- Current mitigation strategies
- Recommended policy adjustments
- Portfolio optimization opportunities
- Monitoring enhancements

## STRATEGIC RECOMMENDATIONS
- Risk appetite considerations
- Business strategy alignment
- Capital allocation guidance
- Performance improvement initiatives

Please provide insights that would be valuable for executive decision-making and board oversight."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def model_explanation_prompt(
        prediction_details: Dict[str, Any],
        feature_contributions: List[Dict[str, Any]],
        borrower_profile: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate prompt for model prediction explanation.
        
        Args:
            prediction_details: Detailed prediction results
            feature_contributions: SHAP values or feature contributions
            borrower_profile: Borrower characteristics
        
        Returns:
            List of messages for LLM chat completion
        """
        system_prompt = """You are a credit analyst specializing in model interpretability and explainable AI. Your role is to translate complex model predictions into clear, understandable explanations for loan officers, underwriters, and borrowers.

Explanation principles:
- Clear, non-technical language
- Focus on business drivers
- Actionable insights
- Fair and unbiased interpretation
- Regulatory compliance (fair lending)"""

        prediction_summary = f"""
PREDICTION DETAILS:
- Probability of Default: {prediction_details.get('pd', 0):.2%}
- Risk Rating: {prediction_details.get('risk_rating', 'N/A')}
- Model Confidence: {prediction_details.get('confidence', 0):.1%}
- Decision: {prediction_details.get('decision', 'N/A')}
"""

        contributions_summary = "FEATURE CONTRIBUTIONS:\n"
        for feature in feature_contributions[:10]:
            impact = "increases" if feature.get('contribution', 0) > 0 else "decreases"
            contributions_summary += f"- {feature.get('name', 'N/A')}: {impact} risk by {abs(feature.get('contribution', 0)):.3f}\n"

        borrower_summary = f"""
BORROWER PROFILE:
- Credit Score: {borrower_profile.get('credit_score', 'N/A')}
- Annual Income: ${borrower_profile.get('annual_income', 0):,}
- Debt-to-Income: {borrower_profile.get('debt_to_income_ratio', 0):.1%}
- Employment Length: {borrower_profile.get('employment_length', 'N/A')} years
- Loan Purpose: {borrower_profile.get('loan_purpose', 'N/A')}
"""

        user_prompt = f"""Please explain this credit risk model prediction in clear, understandable terms:

{prediction_summary}

{contributions_summary}

{borrower_summary}

Please structure your explanation as follows:

## PREDICTION SUMMARY
- Overall risk assessment in plain language
- Confidence level explanation
- Decision rationale

## KEY FACTORS ANALYSIS
- Most important factors driving the prediction
- How each factor impacts the risk assessment
- Relative importance of different factors

## POSITIVE FACTORS
- Strengths that reduce credit risk
- Protective elements in the profile
- Factors supporting creditworthiness

## RISK FACTORS
- Elements that increase credit risk
- Areas of concern or weakness
- Factors requiring attention

## BUSINESS INTERPRETATION
- What this means for lending decision
- Practical implications for the borrower
- Considerations for loan structuring

## IMPROVEMENT OPPORTUNITIES
- Actions borrower could take to improve profile
- Factors that could change over time
- Monitoring recommendations

## REGULATORY CONSIDERATIONS
- Fair lending compliance notes
- Adverse action explanation elements
- Documentation requirements

Please ensure the explanation is accessible to non-technical users while maintaining accuracy and completeness."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def stress_testing_prompt(
        base_scenario: Dict[str, Any],
        stress_scenarios: List[Dict[str, Any]],
        portfolio_composition: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate prompt for stress testing analysis.
        
        Args:
            base_scenario: Baseline economic scenario
            stress_scenarios: List of stress test scenarios
            portfolio_composition: Portfolio characteristics
        
        Returns:
            List of messages for LLM chat completion
        """
        system_prompt = """You are a stress testing specialist with expertise in credit risk modeling and regulatory stress testing frameworks (CCAR, CECL). Your role is to design comprehensive stress scenarios and interpret stress testing results for risk management and regulatory purposes.

Stress testing principles:
- Regulatory compliance (Fed guidance)
- Severe but plausible scenarios
- Forward-looking perspective
- Portfolio-specific considerations
- Actionable risk insights"""

        base_summary = f"""
BASE SCENARIO:
- GDP Growth: {base_scenario.get('gdp_growth', 'N/A')}%
- Unemployment Rate: {base_scenario.get('unemployment_rate', 'N/A')}%
- Interest Rates: {base_scenario.get('interest_rates', 'N/A')}%
- House Price Index: {base_scenario.get('house_prices', 'N/A')}%
- Stock Market: {base_scenario.get('stock_market', 'N/A')}%
"""

        stress_summary = "STRESS SCENARIOS:\n"
        for i, scenario in enumerate(stress_scenarios, 1):
            stress_summary += f"""
Scenario {i}: {scenario.get('name', 'N/A')}
- GDP Growth: {scenario.get('gdp_growth', 'N/A')}%
- Unemployment: {scenario.get('unemployment_rate', 'N/A')}%
- Expected Loss Impact: {scenario.get('loss_impact', 'N/A')}%
"""

        portfolio_summary = f"""
PORTFOLIO COMPOSITION:
- Total Exposure: ${portfolio_composition.get('total_exposure', 0):,.0f}
- Consumer Loans: {portfolio_composition.get('consumer_pct', 0):.1%}
- Commercial Loans: {portfolio_composition.get('commercial_pct', 0):.1%}
- Real Estate: {portfolio_composition.get('real_estate_pct', 0):.1%}
- Geographic Concentration: {portfolio_composition.get('geographic_concentration', 'N/A')}
"""

        user_prompt = f"""Please provide comprehensive stress testing analysis:

{base_summary}

{stress_summary}

{portfolio_summary}

Please structure your analysis as follows:

## STRESS TESTING OVERVIEW
- Purpose and regulatory context
- Scenario design rationale
- Testing methodology

## SCENARIO ANALYSIS
- Base case assumptions and outlook
- Stress scenario descriptions
- Severity assessment and plausibility

## PORTFOLIO VULNERABILITY ASSESSMENT
- Key risk exposures
- Concentration risks
- Sensitivity analysis

## STRESS TEST RESULTS
- Expected loss projections
- Capital impact assessment
- Liquidity considerations
- Performance vs. thresholds

## RISK DRIVER ANALYSIS
- Primary stress factors
- Correlation effects
- Second-order impacts
- Tail risk considerations

## MANAGEMENT ACTIONS
- Risk mitigation strategies
- Capital planning implications
- Portfolio optimization opportunities
- Contingency planning

## REGULATORY IMPLICATIONS
- CCAR/DFAST compliance
- CECL impact assessment
- Supervisory expectations
- Documentation requirements

## RECOMMENDATIONS
- Risk management enhancements
- Stress testing improvements
- Monitoring recommendations
- Strategic considerations

Please provide insights suitable for senior management and regulatory reporting."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


# Utility functions for prompt customization
def customize_prompt_for_audience(
    base_prompt: List[Dict[str, str]], 
    audience: str = "technical"
) -> List[Dict[str, str]]:
    """
    Customize prompt based on target audience.
    
    Args:
        base_prompt: Base prompt messages
        audience: Target audience ("technical", "business", "regulatory", "executive")
    
    Returns:
        Customized prompt messages
    """
    audience_instructions = {
        "technical": "Focus on technical details, model mechanics, and statistical concepts.",
        "business": "Emphasize business implications, practical applications, and strategic insights.",
        "regulatory": "Highlight compliance considerations, regulatory requirements, and documentation needs.",
        "executive": "Provide high-level strategic perspective with actionable recommendations."
    }
    
    if audience in audience_instructions:
        # Add audience-specific instruction to system prompt
        system_msg = base_prompt[0]["content"]
        system_msg += f"\n\nAUDIENCE FOCUS: {audience_instructions[audience]}"
        base_prompt[0]["content"] = system_msg
    
    return base_prompt


def add_context_to_prompt(
    base_prompt: List[Dict[str, str]], 
    additional_context: str
) -> List[Dict[str, str]]:
    """
    Add additional context to prompt.
    
    Args:
        base_prompt: Base prompt messages
        additional_context: Additional context to include
    
    Returns:
        Enhanced prompt messages
    """
    user_msg = base_prompt[-1]["content"]
    user_msg += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"
    base_prompt[-1]["content"] = user_msg
    
    return base_prompt 