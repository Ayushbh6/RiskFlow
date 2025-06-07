"""
LLM integration package for the credit risk MLOps system.
"""

from .openai_client import (
    OpenAIClient,
    get_openai_client,
    is_openai_available
)

from .ollama_client import (
    OllamaClient,
    get_ollama_client,
    is_ollama_available
)

from .risk_analyzer import (
    RiskAnalyzer,
    get_risk_analyzer,
    is_llm_available
)

from .documentation_generator import (
    DocumentationGenerator,
    get_documentation_generator
)

from .prompt_templates import (
    PromptTemplates,
    customize_prompt_for_audience,
    add_context_to_prompt
)

__all__ = [
    # OpenAI
    "OpenAIClient",
    "get_openai_client", 
    "is_openai_available",
    
    # Ollama
    "OllamaClient",
    "get_ollama_client",
    "is_ollama_available",
    
    # Risk Analyzer
    "RiskAnalyzer",
    "get_risk_analyzer",
    "is_llm_available",
    
    # Documentation Generator
    "DocumentationGenerator",
    "get_documentation_generator",
    
    # Prompt Templates
    "PromptTemplates",
    "customize_prompt_for_audience",
    "add_context_to_prompt"
] 