"""
Integration tests for LLM components.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, Mock

from src.llm.risk_analyzer import RiskAnalyzer, get_risk_analyzer
from src.llm.documentation_generator import DocumentationGenerator
from src.llm.openai_client import OpenAIClient, is_openai_available
from src.llm.ollama_client import OllamaClient, is_ollama_available
from src.utils.exceptions import LLMError, ConfigurationException


class TestLLMIntegration:
    """Integration tests for LLM components."""
    
    def test_risk_analyzer_with_mock_providers(self):
        """Test risk analyzer with mocked LLM providers."""
        with patch('src.llm.risk_analyzer.get_openai_client') as mock_openai, \
             patch('src.llm.risk_analyzer.get_ollama_client') as mock_ollama, \
             patch('src.llm.risk_analyzer.is_openai_available', return_value=True), \
             patch('src.llm.risk_analyzer.is_ollama_available', return_value=True), \
             patch('src.llm.risk_analyzer.get_llm_config', return_value={"provider": "openai"}):
            
            # Mock OpenAI client
            mock_openai_client = Mock()
            mock_openai_client.chat_completion.return_value = {
                "content": "This borrower presents moderate credit risk...",
                "model": "gpt-4.1-mini",
                "usage": {"total_tokens": 150},
                "created_at": "2024-01-01T00:00:00Z",
                "finish_reason": "stop"
            }
            mock_openai.return_value = mock_openai_client
            
            # Mock Ollama client
            mock_ollama_client = Mock()
            mock_ollama.return_value = mock_ollama_client
            
            analyzer = RiskAnalyzer()
            
            borrower_data = {
                "borrower_id": "TEST001",
                "industry": "Technology",
                "annual_revenue": 5000000,
                "credit_score": 720,
                "debt_to_income_ratio": 0.35
            }
            
            model_prediction = {
                "pd": 0.08,
                "lgd": 0.45,
                "expected_loss": 0.036,
                "risk_rating": "B",
                "confidence": 0.82
            }
            
            result = analyzer.analyze_credit_risk(
                borrower_data=borrower_data,
                model_prediction=model_prediction
            )
            
            assert result["borrower_id"] == "TEST001"
            assert result["analysis_type"] == "credit_risk"
            assert "moderate credit risk" in result["analysis"].lower()
            assert "usage" in result
            assert "created_at" in result
    
    def test_risk_analyzer_fallback_mechanism(self):
        """Test fallback from OpenAI to Ollama."""
        with patch('src.llm.risk_analyzer.get_openai_client') as mock_openai, \
             patch('src.llm.risk_analyzer.get_ollama_client') as mock_ollama, \
             patch('src.llm.risk_analyzer.is_openai_available', return_value=True), \
             patch('src.llm.risk_analyzer.is_ollama_available', return_value=True), \
             patch('src.llm.risk_analyzer.get_llm_config', return_value={"provider": "openai"}):
            
            # Mock OpenAI client to fail
            mock_openai_client = Mock()
            mock_openai_client.chat_completion.side_effect = Exception("OpenAI API Error")
            mock_openai.return_value = mock_openai_client
            
            # Mock Ollama client to succeed
            mock_ollama_client = Mock()
            mock_ollama_client.chat_completion.return_value = {
                "content": "Fallback analysis from Ollama...",
                "model": "gemma3:4b",
                "usage": {"total_tokens": 120},
                "created_at": "2024-01-01T00:00:00Z",
                "finish_reason": "stop"
            }
            mock_ollama.return_value = mock_ollama_client
            
            analyzer = RiskAnalyzer()
            
            borrower_data = {"borrower_id": "TEST002"}
            model_prediction = {"pd": 0.05}
            
            result = analyzer.analyze_credit_risk(
                borrower_data=borrower_data,
                model_prediction=model_prediction
            )
            
            assert "fallback analysis" in result["analysis"].lower()
            # Verify OpenAI was tried first
            mock_openai_client.chat_completion.assert_called_once()
            # Verify Ollama was used as fallback
            mock_ollama_client.chat_completion.assert_called_once()
    
    def test_documentation_generator_integration(self):
        """Test documentation generator with risk analyzer."""
        with patch('src.llm.documentation_generator.get_risk_analyzer') as mock_get_analyzer:
            
            # Mock risk analyzer
            mock_analyzer = Mock()
            mock_analyzer.generate_model_documentation.return_value = {
                "documentation": "# Credit Risk Model Documentation\n\nThis model predicts...",
                "metadata": {"model": "gpt-4.1-mini", "provider": "openai"},
                "created_at": "2024-01-01T00:00:00Z"
            }
            mock_get_analyzer.return_value = mock_analyzer
            
            # Model registry is now handled by MockModelRegistry in DocumentationGenerator
            
            doc_gen = DocumentationGenerator()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch.object(doc_gen, 'docs_dir', temp_dir):
                    result = doc_gen.generate_model_documentation(
                        model_id="credit_risk_v2",
                        audience="technical",
                        output_format="markdown"
                    )
                
                assert result["model_id"] == "credit_risk_v2"
                assert result["format"] == "markdown"
                assert "documentation" in result
                assert "formatted_content" in result
                assert "file_path" in result
                
                # Verify file was created
                assert os.path.exists(result["file_path"])
                
                # Verify content
                with open(result["file_path"], 'r') as f:
                    content = f.read()
                    assert "Credit Risk Model Documentation" in content
                    assert "Generated:" in content
                    assert "Provider: openai" in content
    
    def test_risk_report_generation(self):
        """Test risk report generation end-to-end."""
        with patch('src.llm.documentation_generator.get_risk_analyzer') as mock_get_analyzer, \
             patch('src.llm.documentation_generator.get_model_registry') as mock_get_registry:
            
            # Mock risk analyzer
            mock_analyzer = Mock()
            mock_analyzer.analyze_credit_risk.return_value = {
                "analysis": "The borrower shows strong financial metrics...",
                "borrower_id": "REPORT001",
                "analysis_type": "credit_risk",
                "provider_used": "openai",
                "metadata": {"model": "gpt-4.1-mini"},
                "created_at": "2024-01-01T00:00:00Z",
                "usage": {"total_tokens": 200}
            }
            mock_get_analyzer.return_value = mock_analyzer
            
            # Mock model registry (not used in risk reports but required for initialization)
            mock_registry = Mock()
            mock_get_registry.return_value = mock_registry
            
            doc_gen = DocumentationGenerator()
            
            borrower_data = {
                "borrower_id": "REPORT001",
                "company_name": "Tech Innovations Inc.",
                "industry": "Technology",
                "annual_revenue": 10000000
            }
            
            model_prediction = {
                "pd": 0.03,
                "lgd": 0.35,
                "expected_loss": 0.0105,
                "risk_rating": "A-",
                "confidence": 0.91
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch.object(doc_gen, 'docs_dir', temp_dir):
                    result = doc_gen.generate_risk_report(
                        borrower_data=borrower_data,
                        model_prediction=model_prediction,
                        output_format="html"
                    )
                
                assert result["borrower_id"] == "REPORT001"
                assert result["format"] == "html"
                assert "report" in result
                assert "file_path" in result
                
                # Verify file was created
                assert os.path.exists(result["file_path"])
                
                # Verify content
                with open(result["file_path"], 'r') as f:
                    content = f.read()
                    assert "REPORT001" in content
                    assert "Tech Innovations Inc." in content
                    assert "Risk Rating: A-" in content
                    assert "strong financial metrics" in content
    
    def test_multiple_format_generation(self):
        """Test generating documentation in multiple formats."""
        with patch('src.llm.documentation_generator.get_risk_analyzer') as mock_get_analyzer, \
             patch('src.llm.documentation_generator.get_model_registry') as mock_get_registry:
            
            # Mock dependencies
            mock_analyzer = Mock()
            mock_analyzer.analyze_credit_risk.return_value = {
                "analysis": "Multi-format test analysis...",
                "borrower_id": "MULTI001",
                "analysis_type": "credit_risk",
                "provider_used": "openai",
                "metadata": {"model": "gpt-4.1-mini"},
                "created_at": "2024-01-01T00:00:00Z"
            }
            mock_get_analyzer.return_value = mock_analyzer
            
            mock_registry = Mock()
            mock_get_registry.return_value = mock_registry
            
            doc_gen = DocumentationGenerator()
            
            borrower_data = {"borrower_id": "MULTI001"}
            model_prediction = {"pd": 0.05, "risk_rating": "B+"}
            
            formats = ["markdown", "html", "json"]
            results = {}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch.object(doc_gen, 'docs_dir', temp_dir):
                    for fmt in formats:
                        result = doc_gen.generate_risk_report(
                            borrower_data=borrower_data,
                            model_prediction=model_prediction,
                            output_format=fmt
                        )
                        results[fmt] = result
                        
                        # Verify each format
                        assert result["format"] == fmt
                        assert os.path.exists(result["file_path"])
                        
                        # Verify format-specific content
                        with open(result["file_path"], 'r') as f:
                            content = f.read()
                            
                            if fmt == "markdown":
                                assert content.startswith("# Credit Risk Report")
                            elif fmt == "html":
                                assert content.startswith("<!DOCTYPE html>")
                                assert "<html>" in content
                            elif fmt == "json":
                                # Should be valid JSON
                                data = json.loads(content)
                                assert data["title"] == "Credit Risk Report - MULTI001"
                                assert data["format"] == "json"
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        with patch('src.llm.documentation_generator.get_risk_analyzer') as mock_get_analyzer, \
             patch('src.llm.documentation_generator.get_model_registry') as mock_get_registry:
            
            # Mock risk analyzer to raise an error
            mock_analyzer = Mock()
            mock_analyzer.analyze_credit_risk.side_effect = LLMError("LLM service unavailable")
            mock_get_analyzer.return_value = mock_analyzer
            
            mock_registry = Mock()
            mock_get_registry.return_value = mock_registry
            
            doc_gen = DocumentationGenerator()
            
            borrower_data = {"borrower_id": "ERROR001"}
            model_prediction = {"pd": 0.05}
            
            with pytest.raises(LLMError):
                doc_gen.generate_risk_report(
                    borrower_data=borrower_data,
                    model_prediction=model_prediction
                )
    
    def test_configuration_integration(self):
        """Test configuration integration across components."""
        with patch('src.llm.risk_analyzer.get_llm_config') as mock_config, \
             patch('src.llm.risk_analyzer.is_openai_available', return_value=False), \
             patch('src.llm.risk_analyzer.is_ollama_available', return_value=False):
            
            mock_config.return_value = {"provider": "openai"}
            
            # Should raise ConfigurationException when no providers available
            with pytest.raises(ConfigurationException):
                RiskAnalyzer()
    
    def test_caching_integration(self):
        """Test caching behavior across components."""
        with patch('src.llm.risk_analyzer.get_openai_client') as mock_openai, \
             patch('src.llm.risk_analyzer.is_openai_available', return_value=True), \
             patch('src.llm.risk_analyzer.is_ollama_available', return_value=False), \
             patch('src.llm.risk_analyzer.get_llm_config', return_value={"provider": "openai"}):
            
            # Mock OpenAI client with caching
            mock_openai_client = Mock()
            mock_openai_client.chat_completion.return_value = {
                "content": "Cached response...",
                "model": "gpt-4.1-mini",
                "usage": {"total_tokens": 100},
                "created_at": "2024-01-01T00:00:00Z",
                "finish_reason": "stop",
                "cached": True
            }
            mock_openai.return_value = mock_openai_client
            
            analyzer = RiskAnalyzer()
            
            borrower_data = {"borrower_id": "CACHE001"}
            model_prediction = {"pd": 0.05}
            
            # Make the same request twice
            result1 = analyzer.analyze_credit_risk(
                borrower_data=borrower_data,
                model_prediction=model_prediction
            )
            
            result2 = analyzer.analyze_credit_risk(
                borrower_data=borrower_data,
                model_prediction=model_prediction
            )
            
            # Both should return the same content
            assert result1["analysis"] == result2["analysis"]
            
            # OpenAI client should be called for both (caching is handled internally)
            assert mock_openai_client.chat_completion.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__]) 