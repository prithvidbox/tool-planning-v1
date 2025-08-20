"""
RAG-based NLU-to-Tool Matching System

Open-source intent matching system for MCP server tool orchestration.
"""

from .config_parser import ConfigParser, IntentConfig
from .embedding_engine import EmbeddingEngine, RRFEngine
from .intent_matcher import IntentMatcher, ToolExecutor
from .dependency_planner import DependencyPlanner, ToolDependency, SmartToolPlanner
from .lightweight_nlp import LightweightNLP, SmartVariableCollector

__version__ = "1.0.0"
__author__ = "MCP Configuration Generator"

__all__ = [
    "ConfigParser",
    "IntentConfig", 
    "EmbeddingEngine",
    "RRFEngine",
    "IntentMatcher",
    "ToolExecutor",
    "DependencyPlanner",
    "ToolDependency",
    "SmartToolPlanner",
    "LightweightNLP",
    "SmartVariableCollector"
]
