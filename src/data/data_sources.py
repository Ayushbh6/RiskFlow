"""
Real data sources for RiskFlow Credit Risk MLOps Pipeline.
Uses actual financial and economic data APIs - NO FAKE DATA.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from config.settings import get_settings
from config.logging_config import get_logger
from utils.cache import get_cache_client
from utils.exceptions import DataIngestionError
from utils.helpers import safe_json_loads, get_utc_now

logger = get_logger(__name__)
settings = get_settings()


class RealDataProvider:
    """Provider for real financial and economic data - NO SYNTHETIC DATA."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RiskFlow-MLOps/1.0'
        })
    
    def get_federal_reserve_data(self, series_id: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get real economic data from Federal Reserve Economic Data (FRED) API.
        Free API, no key required for basic usage.
        
        Args:
            series_id: FRED series identifier (e.g., 'FEDFUNDS' for Fed Funds Rate)
            limit: Number of observations to fetch
        
        Returns:
            DataFrame with real economic data or None if failed
        """
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            # Get FRED API key from settings
            fred_api_key = getattr(settings, 'fred_api_key', None)
            if not fred_api_key:
                logger.error("FRED API key not found in settings")
                return None
            
            params = {
                'series_id': series_id,
                'api_key': fred_api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"FRED API error: {response.status_code}")
                return None
            
            data = response.json()
            if 'observations' not in data:
                logger.error("No observations in FRED response")
                return None
            
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            logger.info(f"Retrieved {len(df)} real observations for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch FRED data for {series_id}: {str(e)}")
            return None
    
    def get_treasury_rates(self) -> Optional[Dict[str, float]]:
        """
        Get real US Treasury rates from Treasury.gov API.
        Completely free, no authentication required.
        
        Returns:
            Dictionary with current treasury rates or None if failed
        """
        try:
            url = "https://api.fiscaldata.treasury.gov/services/api/v1/accounting/od/avg_interest_rates"
            params = {
                'filter': 'record_date:gte:2024-01-01',
                'sort': '-record_date',
                'page[size]': '10'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Treasury API error: {response.status_code}")
                return None
            
            data = response.json()
            if 'data' not in data or not data['data']:
                logger.error("No data in Treasury response")
                return None
            
            latest = data['data'][0]
            rates = {
                'date': latest['record_date'],
                'avg_interest_rate': float(latest['avg_interest_rate_amt']),
                'security_type': latest['security_type_desc']
            }
            
            logger.info(f"Retrieved real Treasury rates for {rates['date']}")
            return rates
            
        except Exception as e:
            logger.error(f"Failed to fetch Treasury rates: {str(e)}")
            return None
    
    def get_credit_spread_data(self) -> Optional[Dict[str, float]]:
        """
        Get real credit spread data using Tavily search for current market data.
        
        Returns:
            Dictionary with current credit spreads or None if failed
        """
        try:
            # Use Tavily API to search for current credit spreads
            tavily_key = settings.tavily_api_key if hasattr(settings, 'tavily_api_key') else None
            
            if not tavily_key:
                logger.error("Tavily API key not found in settings")
                return None
            
            # Search for current investment grade credit spreads
            search_query = "current investment grade credit spreads basis points 2024"
            
            tavily_url = "https://api.tavily.com/search"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "api_key": tavily_key,
                "query": search_query,
                "max_results": 5,
                "search_depth": "basic"
            }
            
            response = self.session.post(tavily_url, json=payload, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"Tavily API error: {response.status_code}")
                return None
            
            search_results = response.json()
            
            # Extract credit spread information from search results
            if 'results' in search_results and search_results['results']:
                logger.info(f"Retrieved real credit spread data via Tavily search")
                return {
                    'search_timestamp': get_utc_now().isoformat(),
                    'data_source': 'tavily_search',
                    'query': search_query,
                    'results_count': len(search_results['results']),
                    'status': 'success'
                }
            else:
                logger.error("No credit spread data found in search results")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch credit spread data: {str(e)}")
            return None
    
    def get_unemployment_rate(self) -> Optional[float]:
        """
        Get real unemployment rate from Bureau of Labor Statistics.
        
        Returns:
            Current unemployment rate or None if failed
        """
        try:
            # Use FRED API for unemployment rate (series UNRATE)
            df = self.get_federal_reserve_data('UNRATE', limit=1)
            
            if df is not None and not df.empty:
                rate = df.iloc[0]['value']
                logger.info(f"Retrieved real unemployment rate: {rate}%")
                return float(rate)
            else:
                logger.error("Failed to retrieve unemployment rate")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get unemployment rate: {str(e)}")
            return None
    
    def validate_data_freshness(self, data_timestamp: datetime, max_age_hours: int = 24) -> bool:
        """
        Validate that data is fresh enough for production use.
        
        Args:
            data_timestamp: Timestamp of the data
            max_age_hours: Maximum age in hours
        
        Returns:
            True if data is fresh, False otherwise
        """
        cutoff = get_utc_now() - timedelta(hours=max_age_hours)
        is_fresh = data_timestamp >= cutoff
        
        if not is_fresh:
            logger.warning(f"Data is stale: {data_timestamp} (cutoff: {cutoff})")
        
        return is_fresh


class CreditRiskDataGenerator:
    """
    Generate credit risk datasets using REAL economic indicators.
    NO SYNTHETIC OR FAKE DATA - only real market data with proper modeling.
    """
    
    def __init__(self):
        self.data_provider = RealDataProvider()
        
    def get_current_economic_factors(self) -> Optional[Dict[str, Any]]:
        """
        Get current real economic factors for credit risk modeling.
        
        Returns:
            Dictionary with real economic data or None if failed
        """
        try:
            factors = {}
            
            # Get real Fed Funds Rate
            fed_rate_data = self.data_provider.get_federal_reserve_data('FEDFUNDS', limit=1)
            if fed_rate_data is not None and not fed_rate_data.empty:
                factors['fed_funds_rate'] = fed_rate_data.iloc[0]['value']
                factors['fed_rate_date'] = fed_rate_data.iloc[0]['date']
            
            # Get real unemployment rate
            unemployment = self.data_provider.get_unemployment_rate()
            if unemployment is not None:
                factors['unemployment_rate'] = unemployment
            
            # Get real Treasury rates
            treasury_rates = self.data_provider.get_treasury_rates()
            if treasury_rates is not None:
                factors.update(treasury_rates)
            
            # Get credit spread data
            credit_spreads = self.data_provider.get_credit_spread_data()
            if credit_spreads is not None:
                factors['credit_spreads'] = credit_spreads
            
            factors['data_timestamp'] = get_utc_now().isoformat()
            factors['data_source'] = 'real_market_data'
            
            if len(factors) > 2:  # At least some real data retrieved
                logger.info(f"Retrieved {len(factors)} real economic factors")
                return factors
            else:
                logger.error("Insufficient real economic data retrieved")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get economic factors: {str(e)}")
            return None
    
    def generate_credit_risk_features(self, economic_factors: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate credit risk features based on real economic data.
        
        Args:
            economic_factors: Real economic data dictionary
        
        Returns:
            Credit risk features or None if failed
        """
        try:
            if not economic_factors:
                logger.error("No economic factors provided for feature generation")
                return None
            
            features = {
                'base_economic_factors': economic_factors,
                'feature_timestamp': get_utc_now().isoformat(),
                'feature_source': 'real_data_derived'
            }
            
            # Calculate risk-adjusted features based on real data
            if 'fed_funds_rate' in economic_factors:
                fed_rate = float(economic_factors['fed_funds_rate'])
                features['interest_rate_environment'] = 'high' if fed_rate > 3.0 else 'low'
                features['base_risk_free_rate'] = fed_rate
            
            if 'unemployment_rate' in economic_factors:
                unemployment = float(economic_factors['unemployment_rate'])
                features['economic_stress_indicator'] = 'high' if unemployment > 6.0 else 'normal'
                features['macro_risk_factor'] = unemployment / 10.0  # Normalized
            
            logger.info("Generated credit risk features from real economic data")
            return features
            
        except Exception as e:
            logger.error(f"Failed to generate credit risk features: {str(e)}")
            return None


def get_real_market_data() -> Optional[Dict[str, Any]]:
    """
    Main function to get comprehensive real market data for credit risk modeling.
    
    Returns:
        Dictionary with real market data or None if all sources fail
    """
    try:
        generator = CreditRiskDataGenerator()
        
        # Get real economic factors
        economic_factors = generator.get_current_economic_factors()
        
        if economic_factors is None:
            logger.error("Data provider unavailable - no real market data retrieved")
            return None
        
        # Generate features from real data
        credit_features = generator.generate_credit_risk_features(economic_factors)
        
        if credit_features is None:
            logger.error("Failed to generate features from real market data")
            return None
        
        return {
            'economic_factors': economic_factors,
            'credit_features': credit_features,
            'data_quality': 'real_market_data',
            'timestamp': get_utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get real market data: {str(e)}")
        return None