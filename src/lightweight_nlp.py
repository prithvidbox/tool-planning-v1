"""
Simple variable collector for OpenAI-based system.
"""

import re
import os
from typing import Dict, List, Any, Optional
from loguru import logger


class SimpleVariableCollector:
    """Simple variable collector using regex patterns."""
    
    def extract_variables_from_sentence(self, sentence: str, needed_vars: List[str]) -> Dict[str, Any]:
        """
        Extract variables from a natural sentence using regex patterns (fallback only).
        
        Args:
            sentence: Natural language sentence from user
            needed_vars: List of variable names to extract
            
        Returns:
            Dictionary of extracted variables
        """
        variables = {}
        
        # Simple regex patterns for common variables
        patterns = {
            'issue_key': r'\b([A-Z]+-\d+)\b',
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'project': r'\b([A-Z]{2,10})\b',
            'amount': r'\$([0-9,]+(?:\.[0-9]{2})?)',
            'status': r'to\s+([\w\s]+)(?:\s|$)',
            'priority': r'(high|medium|low|critical|urgent)\s*priority',
            'summary': r'"([^"]+)"',
            'comment': r'comment:\s*"([^"]+)"'
        }
        
        for var_name in needed_vars:
            if var_name in patterns:
                matches = re.findall(patterns[var_name], sentence, re.IGNORECASE)
                if matches:
                    variables[var_name] = matches[0]
                    logger.debug(f"Extracted {var_name}: {matches[0]}")
        
        return variables


class SmartVariableCollector:
    """OpenAI-powered variable collector for natural interaction."""
    
    def __init__(self):
        self.simple_collector = SimpleVariableCollector()
        
        # Initialize OpenAI extractor
        try:
            if os.getenv('OPENAI_API_KEY'):
                from .openai_variable_extractor import OpenAIVariableExtractor
                self.openai_extractor = OpenAIVariableExtractor()
                logger.info("Using OpenAI for variable extraction")
            else:
                self.openai_extractor = None
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI extractor: {e}")
            self.openai_extractor = None
    
    def collect_missing_variables(self, missing_vars: List[str], 
                                intent_description: str,
                                intent_examples: List[str],
                                tool_plan: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Collect missing variables using OpenAI or interactive prompts.
        
        Args:
            missing_vars: List of missing variable names
            intent_description: What the intent does
            intent_examples: Example queries for this intent
            tool_plan: Tools that will be executed (for context)
            
        Returns:
            Dictionary of collected variables
        """
        from rich.console import Console
        from rich.prompt import Prompt
        
        console = Console()
        collected = {}
        
        # Show what we need
        console.print(f"[yellow]I need: {', '.join(missing_vars)}[/yellow]")
        
        # Show example
        if intent_examples:
            example = intent_examples[0].replace('{', '').replace('}', '')
            console.print(f"[dim]Example: {example}[/dim]")
        
        # Ask user to provide all missing info in one sentence
        console.print(f"[cyan]Please provide the missing information:[/cyan]")
        user_response = Prompt.ask("Details")
        
        if user_response.strip():
            # Try OpenAI extraction first
            if self.openai_extractor:
                try:
                    extracted = self.openai_extractor.extract_variables_from_query(
                        user_response, missing_vars, intent_description, intent_examples
                    )
                    console.print(f"[green]✅ Understood (via OpenAI):[/green] {extracted}")
                    collected.update(extracted)
                except Exception as e:
                    logger.warning(f"OpenAI extraction failed: {e}, using simple extraction")
                    extracted = self.simple_collector.extract_variables_from_sentence(user_response, missing_vars)
                    console.print(f"[green]✅ Understood:[/green] {extracted}")
                    collected.update(extracted)
            else:
                extracted = self.simple_collector.extract_variables_from_sentence(user_response, missing_vars)
                console.print(f"[green]✅ Understood:[/green] {extracted}")
                collected.update(extracted)
            
            # Ask for any remaining missing variables individually
            still_missing = [var for var in missing_vars if var not in collected]
            if still_missing:
                console.print(f"[yellow]Still need:[/yellow] {', '.join(still_missing)}")
                
                for var in still_missing:
                    try:
                        from cli import get_variable_prompt
                        prompt = get_variable_prompt(var, {})
                        value = Prompt.ask(prompt)
                        if value.strip():
                            collected[var] = value.strip()
                    except (EOFError, KeyboardInterrupt):
                        console.print(f"[yellow]Skipping {var}...[/yellow]")
                        break
        
        return collected
