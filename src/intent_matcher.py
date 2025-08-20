"""
Runtime intent matching and tool execution orchestration.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from .config_parser import ConfigParser, IntentConfig
from .embedding_engine import EmbeddingEngine
from .dependency_planner import DependencyPlanner


class ToolExecutor:
    """Handles tool execution and variable replacement."""
    
    @staticmethod
    def replace_variables(params: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Replace variable placeholders in tool parameters."""
        replaced = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$'):
                var_name = value[1:]  # Remove $ prefix
                replaced[key] = variables.get(var_name)
            elif isinstance(value, str) and '||' in value:
                # Handle default values: $variable || "default"
                parts = value.split('||')
                if len(parts) == 2:
                    var_part = parts[0].strip()
                    default_part = parts[1].strip().strip('"\'')
                    
                    if var_part.startswith('$'):
                        var_name = var_part[1:]
                        replaced[key] = variables.get(var_name, default_part)
                    else:
                        replaced[key] = value
                else:
                    replaced[key] = value
            else:
                replaced[key] = value
                
        return replaced
    
    @staticmethod
    def build_tool_plan(intent_config: IntentConfig, variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build executable tool plan with variables replaced."""
        tool_plan = []
        
        for tool_step in intent_config.tool_plan:
            tool_name = tool_step.get('tool')
            raw_params = tool_step.get('params', {})
            
            # Replace variables in parameters
            params = ToolExecutor.replace_variables(raw_params, variables)
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            tool_plan.append({
                'tool': tool_name,
                'params': params,
                'post_process': tool_step.get('post_process'),
                'note': tool_step.get('note')
            })
            
        return tool_plan


class IntentMatcher:
    """Main class for intent matching and tool orchestration."""
    
    def __init__(self, 
                 jira_config_path: str,
                 hubspot_config_path: str,
                 confidence_threshold: float = 0.8,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the intent matcher.
        
        Args:
            jira_config_path: Path to Jira intent configuration YAML
            hubspot_config_path: Path to HubSpot intent configuration YAML  
            confidence_threshold: Minimum similarity score for intent matching
            model_name: Sentence transformer model name
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.config_parser = ConfigParser()
        self.embedding_engine = EmbeddingEngine(model_name)
        self.variable_extractor = None  # Using lightweight_nlp for variable extraction
        self.tool_executor = ToolExecutor()
        
        # Load configurations
        logger.info("Loading intent configurations")
        self.config_parser.load_all_configs(jira_config_path, hubspot_config_path)
        
        # Build embedding index
        logger.info("Building embedding index")
        self.embedding_engine.build_index(self.config_parser)
        
        logger.info(f"Intent matcher initialized with {len(self.config_parser)} intents")
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return tool execution plan or error.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing:
            - success: bool
            - intent: matched intent name (if successful)
            - platform: platform (jira/hubspot) (if successful)
            - confidence: similarity score (if successful)
            - variables: extracted variables (if successful)
            - tool_plan: executable tool plan (if successful)
            - error: error message (if unsuccessful)
            - suggestions: suggested intents (if unsuccessful)
        """
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Find best matching intent
            match_result = self.embedding_engine.find_best_intent(query, self.confidence_threshold)
            
            if match_result is None:
                # No high-confidence match found
                suggestions = self._get_suggestions(query)
                return {
                    'success': False,
                    'error': 'No high-confidence intent match found',
                    'confidence_threshold': self.confidence_threshold,
                    'suggestions': suggestions
                }
                
            intent_config, confidence = match_result
            
            # Extract variables from the original query first
            from .lightweight_nlp import LightweightNLP
            nlp_extractor = LightweightNLP()
            
            required_vars = [var['name'] for var in intent_config.variables if var.get('required', False)]
            optional_vars = [var['name'] for var in intent_config.variables if not var.get('required', False)]
            all_needed_vars = required_vars + optional_vars
            
            # Try to extract variables from original query
            extracted_vars = nlp_extractor.extract_variables_from_sentence(query, all_needed_vars)
            logger.info(f"Extracted from original query: {extracted_vars}")
            
            # Check which required variables are still missing
            missing_required = []
            for var_name in required_vars:
                if var_name not in extracted_vars:
                    missing_required.append(var_name)
                    
            if missing_required:
                # Return missing variables for CLI to handle
                return {
                    'success': False,
                    'error': f'Missing required variables: {", ".join(missing_required)}',
                    'intent': intent_config.intent,
                    'platform': intent_config.platform,
                    'confidence': confidence,
                    'missing_variables': missing_required,
                    'extracted_variables': extracted_vars,
                    'description': intent_config.description
                }
            else:
                # All required variables available - generate tool plan
                planning_result = DependencyPlanner.plan_tool_execution(intent_config, extracted_vars)
                
                if not planning_result['success']:
                    return {
                        'success': False,
                        'error': planning_result['error'],
                        'intent': intent_config.intent,
                        'platform': intent_config.platform,
                        'confidence': confidence,
                        'missing_variables': planning_result.get('missing_variables', []),
                        'available_variables': planning_result.get('available_variables', [])
                    }
                
                return {
                    'success': True,
                    'intent': intent_config.intent,
                    'platform': intent_config.platform,
                    'confidence': confidence,
                    'variables': extracted_vars,
                    'tool_plan': planning_result['tool_plan'],
                    'execution_order': planning_result['execution_order'],
                    'dependency_analysis': planning_result['dependency_analysis'],
                    'description': intent_config.description
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'error': f'Internal error: {str(e)}'
            }
            
    def _get_suggestions(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get suggested intents for failed matches."""
        try:
            similar_intents = self.embedding_engine.search_similar_intents(query, top_k)
            suggestions = []
            
            for intent_config, score in similar_intents:
                suggestions.append({
                    'intent': intent_config.intent,
                    'platform': intent_config.platform,
                    'description': intent_config.description,
                    'confidence': score,
                    'examples': intent_config.examples[:2]  # First 2 examples
                })
                
            return suggestions
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []
            
    def save_index(self, index_dir: str = "./models") -> None:
        """Save the embedding index to disk."""
        from pathlib import Path
        index_dir = Path(index_dir)
        index_dir.mkdir(exist_ok=True)
        
        index_path = str(index_dir / "intent_index.faiss")
        metadata_path = str(index_dir / "intent_metadata.pkl")
        
        self.embedding_engine.save_index(index_path, metadata_path)
        
    def load_index(self, index_dir: str = "./models") -> None:
        """Load the embedding index from disk."""
        from pathlib import Path
        index_dir = Path(index_dir)
        
        index_path = str(index_dir / "intent_index.faiss")
        metadata_path = str(index_dir / "intent_metadata.pkl")
        
        self.embedding_engine.load_index(index_path, metadata_path)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'config_stats': {
                'total_intents': len(self.config_parser),
                'platforms': self.config_parser.platforms
            },
            'embedding_stats': self.embedding_engine.get_index_stats(),
            'matching_config': {
                'confidence_threshold': self.confidence_threshold,
                'semantic_extraction': True
            }
        }
