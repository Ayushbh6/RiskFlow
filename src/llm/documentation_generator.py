"""
Automated documentation generator for credit risk models and systems.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from .risk_analyzer import get_risk_analyzer
# from ..models.credit_risk_model import get_model_registry  # TODO: Implement when model registry is ready
from ..utils.exceptions import LLMError, DocumentationError
from ..utils.helpers import get_utc_now

logger = logging.getLogger(__name__)


class MockModelRegistry:
    """Mock model registry for testing purposes."""
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_id": model_id,
            "name": f"Credit Risk Model {model_id}",
            "version": "1.0.0",
            "algorithm": "Random Forest",
            "created_at": "2024-01-01T00:00:00Z"
        }
    
    def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get mock model metrics."""
        return {
            "auc_roc": 0.85,
            "accuracy": 0.82,
            "precision": 0.78,
            "recall": 0.80,
            "f1_score": 0.79
        }
    
    def get_feature_importance(self, model_id: str) -> List[Dict[str, Any]]:
        """Get mock feature importance."""
        return [
            {"name": "credit_score", "importance": 0.25},
            {"name": "debt_to_income_ratio", "importance": 0.20},
            {"name": "annual_revenue", "importance": 0.15},
            {"name": "employment_length", "importance": 0.12},
            {"name": "loan_amount", "importance": 0.10}
        ]


class DocumentationGenerator:
    """
    Automated documentation generator for credit risk models and systems.
    
    Generates comprehensive documentation including:
    - Model documentation
    - API documentation
    - Risk analysis reports
    - System architecture documentation
    """
    
    def __init__(self):
        """Initialize documentation generator."""
        self.risk_analyzer = get_risk_analyzer()
        # self.model_registry = get_model_registry()  # TODO: Implement when model registry is ready
        self.model_registry = MockModelRegistry()  # Placeholder for testing
        
        logger.info("DocumentationGenerator initialized")
    
    def generate_model_documentation(
        self,
        model_id: str,
        audience: str = "technical",
        output_format: str = "markdown",
        save_to_file: bool = True,
        output_dir: str = "./docs/models"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model documentation.
        
        Args:
            model_id: Model identifier
            audience: Target audience ("technical", "business", "regulatory")
            output_format: Output format ("markdown", "html", "json")
            save_to_file: Whether to save documentation to file
            output_dir: Directory to save documentation
        
        Returns:
            Dictionary containing generated documentation
        """
        try:
            # Get model information
            model_info = self.model_registry.get_model_info(model_id)
            if not model_info:
                raise DocumentationError(f"Model {model_id} not found in registry")
            
            # Get performance metrics
            performance_metrics = self.model_registry.get_model_metrics(model_id)
            
            # Get feature importance
            feature_importance = self.model_registry.get_feature_importance(model_id)
            
            # Generate documentation using LLM
            doc_result = self.risk_analyzer.generate_model_documentation(
                model_info=model_info,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance,
                audience=audience
            )
            
            # Format documentation
            formatted_doc = self._format_documentation(
                content=doc_result["documentation"],
                title=f"Model Documentation: {model_info.get('name', model_id)}",
                format_type=output_format,
                metadata=doc_result["metadata"]
            )
            
            # Save to file if requested
            if save_to_file:
                file_path = self._save_documentation(
                    content=formatted_doc,
                    filename=f"model_{model_id}_{audience}",
                    format_type=output_format,
                    output_dir=output_dir
                )
                doc_result["file_path"] = file_path
            
            doc_result["formatted_content"] = formatted_doc
            doc_result["format"] = output_format
            
            logger.info(f"Model documentation generated for {model_id}")
            return doc_result
            
        except Exception as e:
            logger.error(f"Failed to generate model documentation: {e}")
            raise DocumentationError(f"Model documentation generation failed: {str(e)}")
    
    def generate_api_documentation(
        self,
        endpoints: List[Dict[str, Any]],
        output_format: str = "markdown",
        save_to_file: bool = True,
        output_dir: str = "./docs/api"
    ) -> Dict[str, Any]:
        """
        Generate API documentation.
        
        Args:
            endpoints: List of API endpoint information
            output_format: Output format ("markdown", "html", "json")
            save_to_file: Whether to save documentation to file
            output_dir: Directory to save documentation
        
        Returns:
            Dictionary containing generated API documentation
        """
        try:
            # Create API documentation content
            api_doc = self._generate_api_content(endpoints)
            
            # Format documentation
            formatted_doc = self._format_documentation(
                content=api_doc,
                title="Credit Risk API Documentation",
                format_type=output_format,
                metadata={"generated_at": get_utc_now().isoformat()}
            )
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                file_path = self._save_documentation(
                    content=formatted_doc,
                    filename="api_documentation",
                    format_type=output_format,
                    output_dir=output_dir
                )
            
            result = {
                "documentation": api_doc,
                "formatted_content": formatted_doc,
                "format": output_format,
                "file_path": file_path,
                "created_at": get_utc_now().isoformat(),
                "endpoints_count": len(endpoints)
            }
            
            logger.info("API documentation generated")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate API documentation: {e}")
            raise DocumentationError(f"API documentation generation failed: {str(e)}")
    
    def generate_risk_report(
        self,
        borrower_data: Dict[str, Any],
        model_prediction: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
        audience: str = "business",
        output_format: str = "markdown",
        save_to_file: bool = True,
        output_dir: str = "./docs/reports"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk analysis report.
        
        Args:
            borrower_data: Borrower information
            model_prediction: Model prediction results
            market_context: Market conditions
            audience: Target audience
            output_format: Output format
            save_to_file: Whether to save to file
            output_dir: Output directory
        
        Returns:
            Dictionary containing generated risk report
        """
        try:
            # Generate risk analysis
            analysis_result = self.risk_analyzer.analyze_credit_risk(
                borrower_data=borrower_data,
                model_prediction=model_prediction,
                market_context=market_context,
                audience=audience
            )
            
            # Create report header
            report_header = self._create_report_header(
                borrower_data=borrower_data,
                model_prediction=model_prediction,
                analysis_result=analysis_result
            )
            
            # Combine header and analysis
            full_report = f"{report_header}\n\n{analysis_result['analysis']}"
            
            # Format documentation
            formatted_doc = self._format_documentation(
                content=full_report,
                title=f"Credit Risk Report: {borrower_data.get('borrower_id', 'Unknown')}",
                format_type=output_format,
                metadata=analysis_result["metadata"]
            )
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                borrower_id = borrower_data.get("borrower_id", "unknown")
                timestamp = get_utc_now().strftime("%Y%m%d_%H%M%S")
                filename = f"risk_report_{borrower_id}_{timestamp}"
                
                file_path = self._save_documentation(
                    content=formatted_doc,
                    filename=filename,
                    format_type=output_format,
                    output_dir=output_dir
                )
            
            result = {
                "report": full_report,
                "formatted_content": formatted_doc,
                "format": output_format,
                "file_path": file_path,
                "borrower_id": borrower_data.get("borrower_id"),
                "analysis_result": analysis_result,
                "created_at": get_utc_now().isoformat()
            }
            
            logger.info(f"Risk report generated for borrower {borrower_data.get('borrower_id')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate risk report: {e}")
            raise DocumentationError(f"Risk report generation failed: {str(e)}")
    
    def generate_system_documentation(
        self,
        system_info: Dict[str, Any],
        output_format: str = "markdown",
        save_to_file: bool = True,
        output_dir: str = "./docs/system"
    ) -> Dict[str, Any]:
        """
        Generate system architecture documentation.
        
        Args:
            system_info: System information and configuration
            output_format: Output format
            save_to_file: Whether to save to file
            output_dir: Output directory
        
        Returns:
            Dictionary containing generated system documentation
        """
        try:
            # Create system documentation content
            system_doc = self._generate_system_content(system_info)
            
            # Format documentation
            formatted_doc = self._format_documentation(
                content=system_doc,
                title="Credit Risk MLOps System Documentation",
                format_type=output_format,
                metadata={"generated_at": get_utc_now().isoformat()}
            )
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                file_path = self._save_documentation(
                    content=formatted_doc,
                    filename="system_documentation",
                    format_type=output_format,
                    output_dir=output_dir
                )
            
            result = {
                "documentation": system_doc,
                "formatted_content": formatted_doc,
                "format": output_format,
                "file_path": file_path,
                "created_at": get_utc_now().isoformat()
            }
            
            logger.info("System documentation generated")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate system documentation: {e}")
            raise DocumentationError(f"System documentation generation failed: {str(e)}")
    
    def _format_documentation(
        self,
        content: str,
        title: str,
        format_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Format documentation content based on output format."""
        if format_type.lower() == "markdown":
            return self._format_markdown(content, title, metadata)
        elif format_type.lower() == "html":
            return self._format_html(content, title, metadata)
        elif format_type.lower() == "json":
            return self._format_json(content, title, metadata)
        else:
            return content
    
    def _format_markdown(self, content: str, title: str, metadata: Dict[str, Any]) -> str:
        """Format content as Markdown."""
        header = f"""# {title}

**Generated:** {get_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Provider:** {metadata.get('model', 'Unknown')}

---

"""
        return header + content
    
    def _format_html(self, content: str, title: str, metadata: Dict[str, Any]) -> str:
        """Format content as HTML."""
        # Convert markdown-style content to basic HTML
        html_content = content.replace('\n## ', '\n<h2>').replace('\n### ', '\n<h3>')
        html_content = html_content.replace('## ', '<h2>').replace('### ', '<h3>')
        html_content = html_content.replace('\n\n', '</p>\n<p>')
        html_content = f"<p>{html_content}</p>"
        html_content = html_content.replace('<h2>', '</p>\n<h2>').replace('<h3>', '</p>\n<h3>')
        html_content = html_content.replace('<h2>', '<h2>').replace('</h2>', '</h2>\n<p>')
        html_content = html_content.replace('<h3>', '<h3>').replace('</h3>', '</h3>\n<p>')
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .metadata {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="metadata">
        <strong>Generated:</strong> {get_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
        <strong>Provider:</strong> {metadata.get('model', 'Unknown')}
    </div>
    <hr>
    {html_content}
</body>
</html>"""
    
    def _format_json(self, content: str, title: str, metadata: Dict[str, Any]) -> str:
        """Format content as JSON."""
        doc_data = {
            "title": title,
            "content": content,
            "metadata": metadata,
            "generated_at": get_utc_now().isoformat(),
            "format": "json"
        }
        return json.dumps(doc_data, indent=2)
    
    def _save_documentation(
        self,
        content: str,
        filename: str,
        format_type: str,
        output_dir: str
    ) -> str:
        """Save documentation to file."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        extensions = {
            "markdown": ".md",
            "html": ".html",
            "json": ".json"
        }
        extension = extensions.get(format_type.lower(), ".txt")
        
        # Create file path
        file_path = output_path / f"{filename}{extension}"
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Documentation saved to {file_path}")
        return str(file_path)
    
    def _generate_api_content(self, endpoints: List[Dict[str, Any]]) -> str:
        """Generate API documentation content."""
        content = """# Credit Risk API Documentation

This document provides comprehensive documentation for the Credit Risk MLOps API endpoints.

## Overview

The Credit Risk API provides endpoints for:
- Credit risk predictions
- Model management
- Risk analysis
- System monitoring

## Authentication

All API endpoints require authentication using API keys passed in the `X-API-Key` header.

## Endpoints

"""
        
        for endpoint in endpoints:
            content += f"""### {endpoint.get('method', 'GET')} {endpoint.get('path', '/')}

**Description:** {endpoint.get('description', 'No description available')}

**Parameters:**
"""
            
            # Add parameters
            params = endpoint.get('parameters', [])
            if params:
                for param in params:
                    content += f"- `{param.get('name')}` ({param.get('type', 'string')}): {param.get('description', 'No description')}\n"
            else:
                content += "- No parameters\n"
            
            # Add example request
            if endpoint.get('example_request'):
                content += f"""
**Example Request:**
```json
{json.dumps(endpoint['example_request'], indent=2)}
```
"""
            
            # Add example response
            if endpoint.get('example_response'):
                content += f"""
**Example Response:**
```json
{json.dumps(endpoint['example_response'], indent=2)}
```
"""
            
            content += "\n---\n\n"
        
        return content
    
    def _generate_system_content(self, system_info: Dict[str, Any]) -> str:
        """Generate system documentation content."""
        content = f"""# Credit Risk MLOps System Documentation

## System Overview

{system_info.get('description', 'Credit Risk MLOps system for automated credit risk assessment and model management.')}

## Architecture

### Components

"""
        
        components = system_info.get('components', [])
        for component in components:
            content += f"""#### {component.get('name', 'Unknown Component')}

**Description:** {component.get('description', 'No description available')}
**Technology:** {component.get('technology', 'Not specified')}
**Status:** {component.get('status', 'Unknown')}

"""
        
        content += """## Configuration

### Environment Variables

"""
        
        env_vars = system_info.get('environment_variables', [])
        for var in env_vars:
            content += f"- `{var.get('name')}`: {var.get('description', 'No description')}\n"
        
        content += """
## Deployment

### Requirements

"""
        
        requirements = system_info.get('requirements', [])
        for req in requirements:
            content += f"- {req}\n"
        
        content += """
## Monitoring

### Health Checks

The system provides health check endpoints for monitoring:

- `/health`: Basic health check
- `/health/detailed`: Detailed system status
- `/metrics`: Prometheus metrics

### Logging

Logs are structured and include:
- Request/response logging
- Model prediction logging
- Error tracking
- Performance metrics

"""
        
        return content
    
    def _create_report_header(
        self,
        borrower_data: Dict[str, Any],
        model_prediction: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> str:
        """Create report header with key information."""
        header = f"""# Credit Risk Analysis Report

**Borrower ID:** {borrower_data.get('borrower_id', 'Unknown')}
**Report Date:** {get_utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Analysis Provider:** {analysis_result.get('provider_used', 'Unknown')}

## Summary

**Risk Rating:** {model_prediction.get('risk_rating', 'N/A')}
**Probability of Default:** {model_prediction.get('pd', 0):.2%}
**Expected Loss:** {model_prediction.get('expected_loss', 0):.2%}
**Model Confidence:** {model_prediction.get('confidence', 0):.1%}

---
"""
        return header


# Global documentation generator instance
_doc_generator: Optional[DocumentationGenerator] = None


def get_documentation_generator() -> DocumentationGenerator:
    """Get or create documentation generator instance."""
    global _doc_generator
    if _doc_generator is None:
        _doc_generator = DocumentationGenerator()
    return _doc_generator 