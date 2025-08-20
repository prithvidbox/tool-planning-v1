"""
OpenAI-powered Intent Matching System

Lightweight intent matching system for MCP server tool orchestration using OpenAI.
"""

from .config_parser import ConfigParser, IntentConfig
from .openai_embedding_engine import OpenAIEmbeddingEngine, RRFEngine
from .intent_matcher import IntentMatcher, ToolExecutor
from .dependency_planner import DependencyPlanner, ToolDependency, SmartToolPlanner
from .lightweight_nlp import SmartVariableCollector
from .openai_variable_extractor import OpenAIVariableExtractor

__version__ = "2.0.0"
__author__ = "OpenAI Intent Matcher"

__all__ = [
    "ConfigParser",
    "IntentConfig", 
    "OpenAIEmbeddingEngine",
    "RRFEngine",
    "IntentMatcher",
    "ToolExecutor",
    "DependencyPlanner",
    "ToolDependency",
    "SmartToolPlanner",
    "SmartVariableCollector",
    "OpenAIVariableExtractor"
]
