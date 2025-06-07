"""
Unit tests for LLM integration components.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.llm.openai_client import OpenAIClient, get_openai_client, is_openai_available
from src.llm.ollama_client import OllamaClient, get_ollama_client, is_ollama_available
from src.llm.risk_analyzer import RiskAnalyzer, get_risk_analyzer, is_llm_available
from src.llm.documentation_generator import DocumentationGenerator, get_documentation_generator
from src.llm.prompt_templates import PromptTemplates, customize_prompt_for_audience, add_context_to_prompt
from src.utils.exceptions import LLMError, ConfigurationException, DocumentationError


class TestPromptTemplates:
    """Test prompt template generation."""
    
    def test_risk_analysis_prompt(self):
        """Test risk analysis prompt generation."""
        borrower_data = {
            "borrower_id": "B001",
            "industry": "Technology",
            "annual_revenue": 1000000,
            "credit_score": 750,
            "debt_to_income_ratio": 0.3,
            "current_ratio": 1.5,
            "roe": 0.15
        }
        
        model_prediction = {
            "pd": 0.05,
            "lgd": 0.4,
            "expected_loss": 0.02,
            "risk_rating": "B+",
            "confidence": 0.85,
            "top_risk_factors": ["High leverage", "Industry volatility"]
        }
        
        market_context = {
            "economic_environment": "Stable",
            "interest_rates": "Rising",
            "industry_outlook": "Positive"
        }
        
        messages = PromptTemplates.risk_analysis_prompt(
            borrower_data=borrower_data,
            model_prediction=model_prediction,
            market_context=market_context
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "credit risk analyst" in messages[0]["content"].lower()
        assert "B001" in messages[1]["content"]
        assert "Technology" in messages[1]["content"]
        assert "5.00%" in messages[1]["content"]  # PD formatted as percentage
    
    def test_model_documentation_prompt(self):
        """Test model documentation prompt generation."""
        model_info = {
            "model_id": "credit_risk_v1",
            "model_type": "Classification",
            "algorithm": "Random Forest",
            "version": "1.0.0",
            "num_features": 25
        }
        
        performance_metrics = {
            "auc_roc": 0.85,
            "accuracy": 0.82,
            "precision": 0.78,
            "recall": 0.80,
            "f1_score": 0.79
        }
        
        feature_importance = [
            {"name": "credit_score", "importance": 0.25},
            {"name": "debt_to_income", "importance": 0.20},
            {"name": "annual_revenue", "importance": 0.15}
        ]
        
        messages = PromptTemplates.model_documentation_prompt(
            model_info=model_info,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance
        )
        
        assert len(messages) == 2
        assert "model risk management" in messages[0]["content"].lower()
        assert "credit_risk_v1" in messages[1]["content"]
        assert "0.850" in messages[1]["content"]  # AUC formatted
        assert "credit_score" in messages[1]["content"]
    
    def test_customize_prompt_for_audience(self):
        """Test prompt customization for different audiences."""
        base_prompt = [
            {"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": "Analyze this data."}
        ]
        
        # Test technical audience
        tech_prompt = customize_prompt_for_audience(base_prompt.copy(), "technical")
        assert "technical details" in tech_prompt[0]["content"].lower()
        
        # Test business audience
        biz_prompt = customize_prompt_for_audience(base_prompt.copy(), "business")
        assert "business implications" in biz_prompt[0]["content"].lower()
        
        # Test regulatory audience
        reg_prompt = customize_prompt_for_audience(base_prompt.copy(), "regulatory")
        assert "compliance considerations" in reg_prompt[0]["content"].lower()
    
    def test_add_context_to_prompt(self):
        """Test adding additional context to prompts."""
        base_prompt = [
            {"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": "Analyze this data."}
        ]
        
        additional_context = "Consider recent market volatility."
        
        enhanced_prompt = add_context_to_prompt(base_prompt.copy(), additional_context)
        assert "recent market volatility" in enhanced_prompt[1]["content"].lower()
        assert "additional context" in enhanced_prompt[1]["content"].lower()


class TestOpenAIClient:
    """Test OpenAI client functionality."""
    
    @patch('src.llm.openai_client.OpenAI')
    @patch('src.llm.openai_client.get_openai_config')
    def test_openai_client_initialization(self, mock_config, mock_openai):
        """Test OpenAI client initialization."""
        mock_config.return_value = {
            "api_key": "test-key",
            "model": "gpt-4.1-mini",
            "max_tokens": 2000,
            "temperature": 0.1,
            "timeout": 30.0
        }
        
        client = OpenAIClient()
        
        assert client.config["api_key"] == "test-key"
        assert client.config["model"] == "gpt-4.1-mini"
        mock_openai.assert_called()
    
    @patch('src.llm.openai_client.get_openai_config')
    def test_openai_client_no_api_key(self, mock_config):
        """Test OpenAI client initialization without API key."""
        mock_config.return_value = {"api_key": None}
        
        with pytest.raises(ConfigurationException):
            OpenAIClient()
    
    @patch('src.llm.openai_client.OpenAI')
    @patch('src.llm.openai_client.get_openai_config')
    def test_chat_completion(self, mock_config, mock_openai):
        """Test chat completion functionality."""
        mock_config.return_value = {
            "api_key": "test-key",
            "model": "gpt-4.1-mini",
            "max_tokens": 2000,
            "temperature": 0.1,
            "timeout": 30.0
        }
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4.1-mini"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = OpenAIClient()
        
        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat_completion(messages)
        
        assert result["content"] == "Test response"
        assert result["model"] == "gpt-4.1-mini"
        assert result["usage"]["total_tokens"] == 15
        assert "created_at" in result
    
    @patch('src.llm.openai_client.OpenAI')
    @patch('src.llm.openai_client.get_openai_config')
    def test_chat_completion_error(self, mock_config, mock_openai):
        """Test chat completion error handling."""
        mock_config.return_value = {
            "api_key": "test-key",
            "model": "gpt-4.1-mini",
            "max_tokens": 2000,
            "temperature": 0.1,
            "timeout": 30.0
        }
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client_instance
        
        client = OpenAIClient()
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(LLMError):
            client.chat_completion(messages)


class TestOllamaClient:
    """Test Ollama client functionality."""
    
    @patch('src.llm.ollama_client.Client')
    @patch('src.llm.ollama_client.get_ollama_config')
    def test_ollama_client_initialization(self, mock_config, mock_client):
        """Test Ollama client initialization."""
        mock_config.return_value = {
            "host": "http://localhost:11434",
            "model": "gemma3:4b",
            "timeout": 60.0
        }
        
        client = OllamaClient()
        
        assert client.config["host"] == "http://localhost:11434"
        assert client.config["model"] == "gemma3:4b"
        mock_client.assert_called()
    
    @patch('src.llm.ollama_client.Client')
    @patch('src.llm.ollama_client.get_ollama_config')
    def test_chat_completion(self, mock_config, mock_client):
        """Test Ollama chat completion."""
        mock_config.return_value = {
            "host": "http://localhost:11434",
            "model": "gemma3:4b",
            "timeout": 60.0
        }
        
        # Mock Ollama response
        mock_response = {
            "message": {"content": "Test response"},
            "model": "gemma3:4b",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "eval_duration": 1000000,
            "load_duration": 500000
        }
        
        mock_client_instance = Mock()
        mock_client_instance.chat.return_value = mock_response
        mock_client_instance.list.return_value = {"models": [{"name": "gemma3:4b"}]}
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        
        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat_completion(messages)
        
        assert result["content"] == "Test response"
        assert result["model"] == "gemma3:4b"
        assert result["usage"]["total_tokens"] == 15
        assert "created_at" in result


if __name__ == "__main__":
    pytest.main([__file__]) 