"""
Configuration parser for intent-to-tool mapping YAML files.
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger


class IntentConfig:
    """Represents a single intent configuration."""
    
    def __init__(self, intent_data: Dict[str, Any], platform: str):
        self.intent = intent_data.get('intent')
        self.description = intent_data.get('description', '')
        self.examples = intent_data.get('examples', [])
        self.variables = intent_data.get('variables', [])
        self.tool_plan = intent_data.get('tool_plan', [])
        self.platform = platform
        
    def get_all_text(self) -> str:
        """Get combined text for embedding generation."""
        texts = [self.description] + self.examples
        return " ".join(filter(None, texts))
    
    def __repr__(self):
        return f"IntentConfig(intent='{self.intent}', platform='{self.platform}', examples={len(self.examples)})"


class ConfigParser:
    """Parses and manages intent configuration files."""
    
    def __init__(self):
        self.intents: List[IntentConfig] = []
        self.platforms: List[str] = []
        
    def load_config_file(self, config_path: str, platform: str) -> None:
        """Load intents from a YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        logger.info(f"Loading {platform} config from {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if not isinstance(data, list):
            raise ValueError(f"Expected list of intents in {config_path}")
            
        platform_intents = []
        for intent_data in data:
            if isinstance(intent_data, dict) and 'intent' in intent_data:
                intent_config = IntentConfig(intent_data, platform)
                self.intents.append(intent_config)
                platform_intents.append(intent_config)
                
        if platform not in self.platforms:
            self.platforms.append(platform)
            
        logger.info(f"Loaded {len(platform_intents)} intents for {platform}")
        
    def load_all_configs(self, jira_config: str, hubspot_config: str) -> None:
        """Load both Jira and HubSpot configuration files."""
        self.load_config_file(jira_config, 'jira')
        self.load_config_file(hubspot_config, 'hubspot')
        
    def get_intents_by_platform(self, platform: str) -> List[IntentConfig]:
        """Get all intents for a specific platform."""
        return [intent for intent in self.intents if intent.platform == platform]
        
    def get_intent_by_name(self, intent_name: str, platform: Optional[str] = None) -> Optional[IntentConfig]:
        """Get a specific intent by name and optionally platform."""
        for intent in self.intents:
            if intent.intent == intent_name:
                if platform is None or intent.platform == platform:
                    return intent
        return None
        
    def get_all_intent_texts(self) -> List[str]:
        """Get all intent texts for embedding generation."""
        return [intent.get_all_text() for intent in self.intents]
        
    def get_intent_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all intents."""
        return [
            {
                'intent': intent.intent,
                'platform': intent.platform,
                'description': intent.description,
                'example_count': len(intent.examples),
                'tool_count': len(intent.tool_plan)
            }
            for intent in self.intents
        ]
        
    def __len__(self):
        return len(self.intents)
        
    def __iter__(self):
        return iter(self.intents)
