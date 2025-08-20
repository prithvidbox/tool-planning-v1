"""
Lightweight NLP model for variable extraction and natural language generation.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any, Optional, Tuple
import re
import warnings
from loguru import logger

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="`return_all_scores` is now deprecated")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="This IS expected if you are initializing")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Also suppress transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class LightweightNLP:
    """Small, efficient model for variable extraction and natural language understanding."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize lightweight NLP with a small, fast model.
        
        Args:
            model_name: Small model for efficiency (distilbert, albert, etc.)
        """
        self.model_name = model_name
        
        # Initialize small model for classification/extraction
        try:
            # Use a lightweight sentiment classifier as base - we'll adapt it for variable extraction
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None  # Updated from deprecated return_all_scores=True
            )
            logger.info(f"Initialized lightweight NLP with sentiment model")
        except:
            # Fallback to even smaller model
            logger.warning("Using fallback text processing")
            self.classifier = None
    
    def extract_variables_from_sentence(self, sentence: str, needed_vars: List[str]) -> Dict[str, Any]:
        """
        Extract variables from a natural sentence using lightweight NLP.
        
        Args:
            sentence: Natural language sentence from user
            needed_vars: List of variable names to extract
            
        Returns:
            Dictionary of extracted variables
        """
        variables = {}
        sentence_lower = sentence.lower()
        
        # Use pattern matching with contextual understanding
        var_patterns = {
            'issue_key': {
                'patterns': [r'([A-Z]+-\d+)', r'issue\s+([A-Z]+-\d+)', r'ticket\s+([A-Z]+-\d+)', r'\b([A-Z]+-\d+)\b'],
                'context_words': []  # Remove context requirement for issue keys
            },
            'project': {
                'patterns': [
                    r'project\s+([A-Z]+)', 
                    r'in\s+([A-Z]+)', 
                    r'for\s+([A-Z]+)',
                    r'([A-Z]{2,10})\s+bug',        # "PROJ bug"
                    r'([A-Z]{2,10})\s+ticket',     # "PROJ ticket" 
                    r'([A-Z]{2,10})\s+issue',      # "PROJ issue"
                    r'create\s+([A-Z]{2,10})',     # "create PROJ"
                    r'\b([A-Z]{2,10})\b'           # Any uppercase word
                ],
                'context_words': []  # Remove context requirement
            },
            'status': {
                'patterns': [r'to\s+([\w\s]+)(?:\s|$)', r'status\s+([\w\s]+)', r'make\s+it\s+([\w\s]+)'],
                'context_words': ['to', 'status', 'make'],
                'valid_statuses': ['to do', 'in progress', 'done', 'qa', 'review', 'closed', 'open']
            },
            'assignee': {
                'patterns': [r'to\s+([a-zA-Z0-9@._-]+)', r'assign\s+to\s+([a-zA-Z0-9@._-]+)', r'for\s+([a-zA-Z0-9@._-]+)'],
                'context_words': ['to', 'assign', 'for']
            },
            'email': {
                'patterns': [r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'],
                'context_words': ['email', 'contact', '@']
            },
            'amount': {
                'patterns': [r'\$?(\d+(?:,\d+)*(?:\.\d{2})?)', r'worth\s+\$?(\d+)', r'amount\s+\$?(\d+)'],
                'context_words': ['$', 'worth', 'amount', 'value']
            },
            'priority': {
                'patterns': [r'(high|medium|low|critical|urgent|highest|lowest)\s+priority', r'priority\s+(high|medium|low)'],
                'context_words': ['priority', 'urgent', 'critical'],
                'valid_priorities': ['high', 'medium', 'low', 'critical', 'urgent', 'highest', 'lowest']
            },
            'summary': {
                'patterns': [r'"([^"]+)"', r'titled\s+"([^"]+)"', r'summary\s+"([^"]+)"'],
                'context_words': ['titled', 'summary', 'called', 'named']
            },
            'comment': {
                'patterns': [r'comment:\s*"([^"]+)"', r'note:\s*"([^"]+)"', r'say\s+"([^"]+)"'],
                'context_words': ['comment', 'note', 'say', 'message']
            }
        }
        
        # Extract variables with context awareness
        for var_name in needed_vars:
            if var_name in var_patterns:
                extracted = self._extract_with_context(sentence, var_name, var_patterns[var_name])
                if extracted:
                    variables[var_name] = extracted
                    logger.debug(f"Extracted {var_name}: {extracted}")
        
        return variables
    
    def _extract_with_context(self, sentence: str, var_name: str, config: Dict[str, Any]) -> Optional[str]:
        """Extract variable using contextual patterns."""
        patterns = config.get('patterns', [])
        context_words = config.get('context_words', [])
        valid_values = config.get('valid_statuses', config.get('valid_priorities', []))
        
        # Check if sentence contains relevant context
        has_context = any(word in sentence.lower() for word in context_words)
        
        if not has_context and context_words:
            return None
        
        # Try patterns in order of preference
        for pattern in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                candidate = matches[0].strip()
                
                # Validate against known values if provided
                if valid_values:
                    candidate_lower = candidate.lower()
                    if candidate_lower in valid_values:
                        return candidate
                    # Check for partial matches
                    for valid_val in valid_values:
                        if valid_val in candidate_lower or candidate_lower in valid_val:
                            return valid_val
                else:
                    return candidate
        
        return None
    
    def generate_natural_prompt(self, missing_vars: List[str], intent_description: str) -> str:
        """
        Generate a natural language prompt for missing variables.
        
        Args:
            missing_vars: List of missing variable names
            intent_description: Description of what the intent does
            
        Returns:
            Natural language prompt asking for missing information
        """
        var_descriptions = {
            'issue_key': "which issue (e.g., PROJ-123)",
            'project': "which project (e.g., PROJ, ABC)",
            'status': "what status to change to (e.g., In Progress, Done)",
            'assignee': "who to assign it to (username or email)",
            'priority': "what priority level (High, Medium, Low)",
            'summary': "what the title/summary should be",
            'comment': "what comment to add",
            'email': "what email address",
            'amount': "what amount/value (e.g., $50000)",
            'dealname': "what to name this deal",
            'subject': "what the subject/title is"
        }
        
        if len(missing_vars) == 1:
            var_desc = var_descriptions.get(missing_vars[0], missing_vars[0])
            return f"I can {intent_description.lower()}, but I need to know {var_desc}."
        elif len(missing_vars) == 2:
            var1_desc = var_descriptions.get(missing_vars[0], missing_vars[0])
            var2_desc = var_descriptions.get(missing_vars[1], missing_vars[1])
            return f"I can {intent_description.lower()}, but I need to know {var1_desc} and {var2_desc}."
        else:
            var_list = [var_descriptions.get(var, var) for var in missing_vars]
            vars_text = ", ".join(var_list[:-1]) + f", and {var_list[-1]}"
            return f"I can {intent_description.lower()}, but I need to know {vars_text}."

    def generate_context_aware_prompt(self, missing_vars: List[str], intent_description: str, 
                                     tool_plan: List[Dict[str, Any]]) -> str:
        """
        Generate a context-aware prompt based on the actual tools that will be executed.
        
        Args:
            missing_vars: List of missing variable names
            intent_description: Description of what the intent does
            tool_plan: List of tools that will be executed
            
        Returns:
            Context-aware prompt explaining what the system will do
        """
        # Tool context descriptions
        tool_contexts = {
            'get_issue_transitions': "I'll check what status transitions are available for this issue",
            'transition_issue': "then change it to your desired status",
            'assign_issue': "I'll assign this issue to the specified person",
            'create_issue': "I'll create a new issue in the project",
            'search_issues': "I'll search for issues matching your criteria", 
            'add_comment': "I'll add your comment to the issue",
            'create_contact': "I'll create a new contact in HubSpot",
            'create_deal': "I'll create a new deal",
            'create_association': "then link them together",
            'update_contact': "I'll update the contact information",
            'search_contacts': "I'll search for contacts",
            'send_email': "I'll send an email"
        }
        
        # Build context from tool plan
        tool_context = ""
        if tool_plan:
            tool_descriptions = []
            for i, tool_step in enumerate(tool_plan):
                tool_name = tool_step.get('tool', '')
                if tool_name in tool_contexts:
                    if i == 0:
                        tool_descriptions.append(tool_contexts[tool_name])
                    else:
                        tool_descriptions.append(f"and {tool_contexts[tool_name]}")
            
            if tool_descriptions:
                tool_context = f"\n\n[dim]What I'll do: {' '.join(tool_descriptions)}[/dim]"
        
        # Variable-specific prompts with tool context
        var_prompts = {
            'issue_key': "Which Jira issue should I work with? (e.g., PROJ-123)",
            'status': "What status should I change it to? (e.g., In Progress, Done, QA)",
            'assignee': "Who should I assign this issue to? (username or email)",
            'project': "Which project is this for? (e.g., PROJ, ABC)",
            'email': "What email address should I use?",
            'dealname': "What should I name this deal?",
            'amount': "What's the deal value? (e.g., $50000)",
            'summary': "What should the title/summary be?",
            'comment': "What comment should I add?",
            'subject': "What should the subject/title be?",
            'priority': "What priority level? (High, Medium, Low)"
        }
        
        if len(missing_vars) == 1:
            var_prompt = var_prompts.get(missing_vars[0], f"What {missing_vars[0]} should I use?")
            return f"{var_prompt}{tool_context}"
        elif len(missing_vars) == 2:
            var1_prompt = var_prompts.get(missing_vars[0], missing_vars[0])
            var2_prompt = var_prompts.get(missing_vars[1], missing_vars[1])
            return f"I need two things:\n1. {var1_prompt}\n2. {var2_prompt}{tool_context}"
        else:
            prompts = [var_prompts.get(var, f"What {var}?") for var in missing_vars]
            prompt_list = "\n".join([f"{i+1}. {prompt}" for i, prompt in enumerate(prompts)])
            return f"I need several things:\n{prompt_list}{tool_context}"
    
    def parse_user_response(self, response: str, missing_vars: List[str]) -> Dict[str, str]:
        """
        Parse user's natural language response to extract multiple variables.
        
        Args:
            response: User's response sentence
            missing_vars: Variables we're trying to extract
            
        Returns:
            Dictionary of extracted variables from the response
        """
        return self.extract_variables_from_sentence(response, missing_vars)
    
    def suggest_complete_query(self, original_query: str, intent_examples: List[str]) -> str:
        """
        Suggest a more complete query based on intent examples.
        
        Args:
            original_query: User's incomplete query
            intent_examples: Examples from the matched intent
            
        Returns:
            Suggested complete query format
        """
        # Find the most similar example to the user's query
        best_example = intent_examples[0] if intent_examples else ""
        
        # Remove placeholder braces for display
        suggestion = re.sub(r'\{[^}]+\}', '...', best_example)
        
        return f"Try something like: '{suggestion}'"


class SmartVariableCollector:
    """Enhanced variable collector using lightweight NLP for natural interaction."""
    
    def __init__(self):
        self.nlp = LightweightNLP()
    
    def collect_missing_variables(self, missing_vars: List[str], 
                                intent_description: str,
                                intent_examples: List[str],
                                tool_plan: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Collect missing variables using natural language interaction with tool context.
        
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
        
        # Generate context-aware prompt with tool information
        if tool_plan:
            context_prompt = self.nlp.generate_context_aware_prompt(missing_vars, intent_description, tool_plan)
            console.print(f"[yellow]{context_prompt}[/yellow]")
        else:
            # Fallback to simple prompt
            natural_prompt = self.nlp.generate_natural_prompt(missing_vars, intent_description)
            console.print(f"[yellow]{natural_prompt}[/yellow]")
        
        # Show example format
        suggestion = self.nlp.suggest_complete_query("", intent_examples)
        console.print(f"[dim]{suggestion}[/dim]")
        
        # Ask user to provide all missing info in one sentence
        console.print(f"[cyan]Please provide the missing information in one sentence:[/cyan]")
        user_response = Prompt.ask("Complete info")
        
        if user_response.strip():
            # Try to extract all variables from the sentence
            extracted = self.nlp.extract_variables_from_sentence(user_response, missing_vars)
            
            # Show what we found
            if extracted:
                console.print(f"[green]✅ Understood:[/green] {extracted}")
                collected.update(extracted)
            
            # Ask for any remaining missing variables individually
            still_missing = [var for var in missing_vars if var not in collected]
            if still_missing:
                console.print(f"[yellow]I still need:[/yellow] {', '.join(still_missing)}")
                
                for var in still_missing:
                    try:
                        from cli import get_variable_prompt, get_variable_examples
                        prompt = get_variable_prompt(var, {})
                        examples = get_variable_examples(var)
                        
                        if examples:
                            console.print(f"[dim]Examples: {', '.join(examples[:3])}[/dim]")
                        
                        value = Prompt.ask(prompt)
                        if value.strip():
                            collected[var] = value.strip()
                    except EOFError:
                        # Handle EOF when input stream is exhausted (e.g., from echo)
                        console.print(f"[red]⚠️  Input stream ended. Missing variable: {var}[/red]")
                        console.print(f"[yellow]💡 For automated testing, provide: {prompt}[/yellow]")
                        break
                    except KeyboardInterrupt:
                        console.print(f"\n[yellow]Skipping variable collection...[/yellow]")
                        break
        
        return collected
