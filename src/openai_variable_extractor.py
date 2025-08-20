"""
OpenAI-based Variable Extractor using GPT-3.5-turbo for intelligent variable extraction.
"""

import os
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIVariableExtractor:
    """Advanced variable extraction using OpenAI GPT-3.5-turbo."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        self.last_token_usage = {}
        self.last_timing = {}
        logger.info(f"Initialized OpenAI Variable Extractor with model: {self.model}")
    
    def extract_variables_from_query(self, query: str, needed_vars: List[str], 
                                   intent_description: str = "", 
                                   intent_examples: List[str] = None) -> Dict[str, Any]:
        """
        Extract variables using OpenAI GPT-3.5-turbo exclusively.
        
        Args:
            query: User query string
            needed_vars: List of variable names to extract
            intent_description: Context about what the intent does
            intent_examples: Example queries for this intent
            
        Returns:
            Dictionary of extracted variables
        """
        try:
            # Build the optimized prompt for GPT-3.5
            prompt = self._build_extraction_prompt(query, needed_vars, intent_description, intent_examples)
            
            # Track timing
            start_time = time.time()
            
            # Call OpenAI API using new v1.0+ format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract variables as JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=150  # Optimized token limit
            )
            
            # Track timing and token usage
            end_time = time.time()
            self.last_timing = {
                'variable_extraction_time': end_time - start_time,
                'timestamp': time.time()
            }
            
            if hasattr(response, 'usage') and response.usage:
                self.last_token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'operation': 'variable_extraction'
                }
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response: {content}")
            
            # Extract JSON from response
            variables = self._parse_openai_response(content)
            
            # Validate and clean extracted variables
            cleaned_variables = self._clean_extracted_variables(variables, needed_vars)
            
            logger.info(f"Extracted variables using OpenAI: {cleaned_variables}")
            return cleaned_variables
            
        except Exception as e:
            logger.error(f"OpenAI variable extraction failed: {e}")
            # Fallback to regex-based extraction only on error
            return self._fallback_extraction(query, needed_vars)
    
    def _build_extraction_prompt(self, query: str, needed_vars: List[str], 
                               intent_description: str, intent_examples: List[str] = None) -> str:
        """Build optimized prompt for OpenAI to extract variables."""
        
        # Concise variable hints
        var_hints = {
            'issue_key': 'PROJ-123',
            'project': 'PROJ',
            'status': 'In Progress',
            'assignee': 'user@email.com',
            'priority': 'High/Medium/Low',
            'email': 'user@domain.com',
            'amount': '$50000',
            'dealname': 'Deal Name',
            'firstname': 'FirstName',
            'lastname': 'LastName',
            'company': 'Company Name',
            'jobtitle': 'Job Title'
        }
        
        # Build concise variable list
        var_list = []
        for var in needed_vars:
            hint = var_hints.get(var, 'value')
            var_list.append(f"{var}: {hint}")
        
        # Add one example if available
        example_text = ""
        if intent_examples:
            example = intent_examples[0].replace('{', '').replace('}', '')
            example_text = f"\nExample: {example}"
        
        # Ultra-concise prompt
        prompt = f"""Query: "{query}"

Extract: {', '.join(var_list)}
{example_text}

Return JSON only:"""
        
        return prompt
    
    def _parse_openai_response(self, content: str) -> Dict[str, Any]:
        """Parse OpenAI response to extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try parsing the whole content as JSON
                return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse OpenAI response as JSON: {content}")
            return {}
    
    def _clean_extracted_variables(self, variables: Dict[str, Any], needed_vars: List[str]) -> Dict[str, Any]:
        """Clean and validate extracted variables."""
        cleaned = {}
        
        for var in needed_vars:
            if var in variables:
                value = variables[var]
                
                # Skip null, empty, or "null" string values
                if value is None or value == "null" or value == "" or str(value).lower() == "none":
                    continue
                
                # Clean the value
                cleaned_value = str(value).strip()
                
                # Specific validation for certain variable types
                if var == 'issue_key':
                    if re.match(r'^[A-Z]+-\d+$', cleaned_value):
                        cleaned[var] = cleaned_value
                elif var == 'email':
                    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', cleaned_value):
                        cleaned[var] = cleaned_value
                elif var == 'project':
                    # Validate project code
                    if re.match(r'^[A-Z][A-Za-z0-9]*$', cleaned_value) and len(cleaned_value) <= 15:
                        cleaned[var] = cleaned_value.upper()
                elif var == 'priority':
                    # Standardize priority values
                    priority_map = {
                        'highest': 'Highest', 'high': 'High', 'medium': 'Medium',
                        'low': 'Low', 'lowest': 'Lowest', 'critical': 'High',
                        'urgent': 'High', 'normal': 'Medium', 'minor': 'Low'
                    }
                    standardized = priority_map.get(cleaned_value.lower(), cleaned_value)
                    cleaned[var] = standardized
                elif var == 'status':
                    # Standardize status values
                    status_map = {
                        'todo': 'To Do', 'to do': 'To Do', 'new': 'To Do',
                        'in progress': 'In Progress', 'inprogress': 'In Progress',
                        'done': 'Done', 'complete': 'Done', 'completed': 'Done',
                        'closed': 'Closed', 'resolved': 'Resolved'
                    }
                    standardized = status_map.get(cleaned_value.lower(), cleaned_value)
                    cleaned[var] = standardized
                else:
                    # For other variables, just ensure they're not empty
                    if len(cleaned_value) > 0:
                        cleaned[var] = cleaned_value
        
        return cleaned
    
    def _fallback_extraction(self, query: str, needed_vars: List[str]) -> Dict[str, Any]:
        """Enhanced regex-based extraction for common patterns."""
        variables = {}
        
        # Enhanced regex patterns for common variables
        patterns = {
            'issue_key': r'\b([A-Z]+-\d+)\b',
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'project': r'\b([A-Z]{2,10})\b',
            'amount': r'\$([0-9,]+(?:\.[0-9]{2})?)',
            'status': r'to\s+((?:In\s+Progress|Done|To\s+Do|QA|Review|Closed|Open|Resolved))',
            'priority': r'(High|Medium|Low|Critical|Urgent|Highest|Lowest)(?:\s+priority)?',
            'firstname': r'\b([A-Z][a-z]+)\s+[A-Z][a-z]+(?:\s+at|\s+@|$)',
            'lastname': r'\b[A-Z][a-z]+\s+([A-Z][a-z]+)(?:\s+at|\s+@|$)',
            'company': r'(?:at\s+|from\s+|@\s+)([A-Z][a-zA-Z\s]+?)(?:\s+[a-z@]|$)',
            'summary': r'"([^"]+)"',
            'comment': r'(?:comment|note|message):\s*"([^"]+)"'
        }
        
        for var in needed_vars:
            if var in patterns:
                matches = re.findall(patterns[var], query, re.IGNORECASE)
                if matches:
                    value = matches[0].strip()
                    # Clean up the value
                    if var == 'status':
                        value = value.replace('  ', ' ').title()
                    elif var in ['firstname', 'lastname']:
                        value = value.capitalize()
                    elif var == 'company':
                        value = value.strip()
                    variables[var] = value
        
        if variables:
            logger.info(f"Enhanced regex extraction: {variables}")
        return variables
    
    def get_extraction_confidence(self, query: str, extracted_vars: Dict[str, Any]) -> Dict[str, float]:
        """
        Get confidence scores for extracted variables.
        
        Args:
            query: Original query
            extracted_vars: Variables that were extracted
            
        Returns:
            Dictionary mapping variable names to confidence scores
        """
        confidence_scores = {}
        
        for var_name, value in extracted_vars.items():
            # Base confidence for OpenAI extraction is higher than regex
            confidence = 0.8
            
            # Higher confidence for structured data that matches patterns
            if var_name == 'issue_key' and re.match(r'^[A-Z]+-\d+$', str(value)):
                confidence = 0.95
            elif var_name == 'email' and '@' in str(value):
                confidence = 0.95
            elif var_name == 'project' and str(value).isupper() and len(str(value)) <= 10:
                confidence = 0.9
            elif var_name in ['status', 'priority'] and str(value) in ['High', 'Medium', 'Low', 'To Do', 'In Progress', 'Done']:
                confidence = 0.9
            
            confidence_scores[var_name] = confidence
        
        return confidence_scores
    
    def validate_extracted_variables(self, extracted_vars: Dict[str, Any], 
                                   required_vars: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate extracted variables and return what's missing.
        
        Args:
            extracted_vars: Variables extracted from query
            required_vars: Variables required for the intent
            
        Returns:
            Tuple of (all_required_present, missing_variables)
        """
        missing = []
        for var in required_vars:
            if var not in extracted_vars or not extracted_vars[var]:
                missing.append(var)
        
        return len(missing) == 0, missing
    
    def enhance_with_context(self, query: str, var_name: str, 
                           current_value: Optional[str], intent_examples: List[str]) -> Optional[str]:
        """
        Use OpenAI to enhance variable extraction with context.
        
        Args:
            query: User query
            var_name: Variable to extract
            current_value: Currently extracted value (may be None)
            intent_examples: Examples from the matched intent
            
        Returns:
            Enhanced/corrected variable value
        """
        if current_value:
            return current_value  # Already have a good value
        
        try:
            prompt = f"""
Given this user query: "{query}"
And these example queries: {intent_examples[:2]}

Extract the value for variable "{var_name}" from the user query.
Consider the patterns shown in the examples.

Respond with only the extracted value, or "null" if not found.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a variable extraction assistant. Respond with only the extracted value or 'null'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            if result.lower() not in ['null', 'none', '']:
                return result
                
        except Exception as e:
            logger.error(f"OpenAI context enhancement failed: {e}")
        
        return None
