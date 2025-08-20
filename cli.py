#!/usr/bin/env python3
"""
Command Line Interface for RAG-based Intent Matching System
"""

import argparse
import sys
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from loguru import logger
from dotenv import load_dotenv

# Suppress all warnings for clean output
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", category=UserWarning)

from src.intent_matcher import IntentMatcher

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="<green>{time}</green> | <level>{level: <8}</level> | {message}")
    else:
        # Only show warnings and errors, not INFO
        logger.add(sys.stderr, level="WARNING", format="<level>{level: <8}</level> | {message}")


def build_index_command(args):
    """Build the embedding index from configuration files."""
    console = Console()
    
    with console.status("[bold green]Building embedding index..."):
        matcher = IntentMatcher(
            jira_config_path=args.jira_config,
            hubspot_config_path=args.hubspot_config,
            confidence_threshold=args.threshold,
            model_name=args.model
        )
        
        if args.save:
            matcher.save_index(args.index_dir)
            console.print(f"✅ Index saved to {args.index_dir}")
        
    # Show statistics
    stats = matcher.get_stats()
    
    table = Table(title="Index Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Intents", str(stats['config_stats']['total_intents']))
    table.add_row("Platforms", ", ".join(stats['config_stats']['platforms']))
    table.add_row("Embedding Dimension", str(stats['embedding_stats']['embedding_dimension']))
    table.add_row("Confidence Threshold", str(args.threshold))
    # Use actual model name from the embedding engine instead of CLI argument
    actual_model = stats['embedding_stats'].get('model_name', args.model)
    engine_type = stats['embedding_stats'].get('engine_type', 'sentence-transformers')
    table.add_row("Model", f"{actual_model} ({engine_type})")
    
    console.print(table)


def test_query_command(args):
    """Test a single query against the intent matcher."""
    console = Console()
    
    # Load matcher
    if Path(args.index_dir).exists():
        console.print("[yellow]Loading existing index...[/yellow]")
        matcher = IntentMatcher(
            jira_config_path=args.jira_config,
            hubspot_config_path=args.hubspot_config,
            confidence_threshold=args.threshold,
            model_name=args.model  # Use correct model from environment
        )
        matcher.load_index(args.index_dir)
    else:
        console.print("[yellow]Building new index...[/yellow]")
        matcher = IntentMatcher(
            jira_config_path=args.jira_config,
            hubspot_config_path=args.hubspot_config,
            confidence_threshold=args.threshold,
            model_name=args.model
        )
    
    # Get top 5 similar intents
    top_intents = matcher.embedding_engine.search_similar_intents(args.query, top_k=5)
    
    # Show top 5 matches table
    matches_table = Table(title=f"Top 5 Intent Matches for: '{args.query}'")
    matches_table.add_column("Rank", style="cyan", width=4)
    matches_table.add_column("Intent", style="green", width=25)
    matches_table.add_column("Platform", style="blue", width=10)
    matches_table.add_column("Confidence", style="yellow", width=10)
    matches_table.add_column("Status", style="white", width=12)
    
    for i, (intent_config, confidence) in enumerate(top_intents, 1):
        status = "✅ MATCH" if confidence >= args.threshold else "❌ Below threshold"
        status_style = "green" if confidence >= args.threshold else "red"
        
        matches_table.add_row(
            str(i),
            intent_config.intent,
            intent_config.platform,
            f"{confidence:.3f}",
            f"[{status_style}]{status}[/{status_style}]"
        )
    
    console.print(matches_table)
    console.print()
    
    # Process query for detailed results
    result = matcher.process_query(args.query)
    
    # Handle missing variables using smart natural language collection
    if not result['success'] and 'missing_variables' in result:
        # Show what was already extracted from original query
        if result.get('extracted_variables'):
            console.print(f"[green]✅ Found in your query:[/green] {result['extracted_variables']}")
        
        # Get the matched intent config
        matched_intent = matcher.config_parser.get_intent_by_name(result['intent'], result['platform'])
        if matched_intent:
            # Use smart variable collector for natural sentence-based collection
            from src.lightweight_nlp import SmartVariableCollector
            
            collector = SmartVariableCollector()
            additional_vars = collector.collect_missing_variables(
                missing_vars=result['missing_variables'],
                intent_description=matched_intent.description,
                intent_examples=matched_intent.examples,
                tool_plan=matched_intent.tool_plan
            )
            
            # Always try to execute with whatever variables we have
            console.print("\n[yellow]Retrying with provided information...[/yellow]")
            
            # Start with variables extracted from original query
            complete_vars = result.get('extracted_variables', {})
            # Add any variables collected interactively
            complete_vars.update(additional_vars)
            
            # Add sensible defaults for missing required variables
            for missing_var in result['missing_variables']:
                if missing_var not in complete_vars:
                    defaults = {
                        'summary': 'Task created via assistant',
                        'priority': 'Medium',
                        'status': 'To Do',
                        'assignee': 'unassigned',
                        'description': 'Details to be provided',
                        'subject': 'Item created via assistant',
                        'email': 'user@example.com'
                    }
                    if missing_var in defaults:
                        complete_vars[missing_var] = defaults[missing_var]
                        console.print(f"[yellow]Using default for {missing_var}: {defaults[missing_var]}[/yellow]")
            
            # Use dependency planner with complete variables
            from src.dependency_planner import DependencyPlanner
            final_result = DependencyPlanner.plan_tool_execution(matched_intent, complete_vars)
            
            if final_result['success']:
                result = {
                    'success': True,
                    'intent': result['intent'],
                    'platform': result['platform'], 
                    'confidence': result['confidence'],
                    'variables': complete_vars,
                    'tool_plan': final_result['tool_plan'],
                    'execution_order': final_result['execution_order'],
                    'dependency_analysis': final_result['dependency_analysis'],
                    'description': result.get('description', '')
                }
            elif 'missing_optional_variables' in final_result and final_result['missing_optional_variables']:
                # Handle optional variables - ask user if they want to provide them
                console.print(f"[yellow]Optional variables available:[/yellow] {', '.join(final_result['missing_optional_variables'])}")
                
                if Confirm.ask("Would you like to provide optional details for better results?", default=False):
                    # Collect optional variables
                    optional_vars = collector.collect_missing_variables(
                        missing_vars=final_result['missing_optional_variables'],
                        intent_description=matched_intent.description,
                        intent_examples=matched_intent.examples,
                        tool_plan=matched_intent.tool_plan
                    )
                    complete_vars.update(optional_vars)
                
                # Try again with optional variables
                final_result = DependencyPlanner.plan_tool_execution(matched_intent, complete_vars)
                
                if final_result['success']:
                    result = {
                        'success': True,
                        'intent': result['intent'],
                        'platform': result['platform'], 
                        'confidence': result['confidence'],
                        'variables': complete_vars,
                        'tool_plan': final_result['tool_plan'],
                        'execution_order': final_result['execution_order'],
                        'dependency_analysis': final_result['dependency_analysis'],
                        'description': result.get('description', '')
                    }
                else:
                    result = final_result
            else:
                result = final_result
    
    # Display detailed results for best match
    if result['success']:
        # Show dependency analysis if available
        dependency_info = ""
        if 'dependency_analysis' in result:
            dep = result['dependency_analysis']
            dependency_info = (
                f"\n[bold]Dependency Analysis:[/bold]\n"
                f"• Total Tools: {dep['total_tools']}\n"
                f"• Execution Order: {' → '.join(result['execution_order'])}\n"
                f"• Tools Reordered: {'Yes' if dep['reordered'] else 'No'}\n"
                f"• Tools with Dependencies: {dep['tools_with_dependencies']}\n"
            )
        
        # Show performance metrics if available
        performance_info = ""
        if hasattr(matcher.embedding_engine, 'last_timing') and hasattr(matcher.variable_extractor, 'last_timing'):
            embed_timing = matcher.embedding_engine.last_timing
            var_timing = matcher.variable_extractor.last_timing
            embed_tokens = matcher.embedding_engine.last_token_usage
            var_tokens = matcher.variable_extractor.last_token_usage
            
            total_time = embed_timing.get('total_time', 0) + var_timing.get('variable_extraction_time', 0)
            total_tokens = embed_tokens.get('embedding_tokens', 0) + var_tokens.get('total_tokens', 0)
            
            performance_info = (
                f"\n[bold]Performance Metrics:[/bold]\n"
                f"• Intent Matching: {embed_timing.get('total_time', 0):.3f}s ({embed_tokens.get('embedding_tokens', 0)} tokens)\n"
                f"• Variable Extraction: {var_timing.get('variable_extraction_time', 0):.3f}s ({var_tokens.get('total_tokens', 0)} tokens)\n"
                f"• Total: {total_time:.3f}s ({total_tokens} tokens)\n"
                f"• Cache Hit: {'Yes' if embed_tokens.get('cached', False) else 'No'}\n"
            )
        
        panel = Panel(
            f"[green]✅ Intent Executed[/green]\n\n"
            f"[bold]Intent:[/bold] {result['intent']}\n"
            f"[bold]Platform:[/bold] {result['platform']}\n"
            f"[bold]Confidence:[/bold] {result['confidence']:.3f}\n"
            f"[bold]Description:[/bold] {result['description']}\n\n"
            f"[bold]Variables Extracted:[/bold]\n{json.dumps(result.get('variables', {}), indent=2)}"
            f"{dependency_info}"
            f"{performance_info}\n"
            f"[bold]Tool Plan:[/bold]\n{json.dumps(result.get('tool_plan', []), indent=2)}",
            title="Execution Details"
        )
    else:
        panel = Panel(
            f"[red]❌ No Intent Above Threshold[/red]\n\n"
            f"[bold]Error:[/bold] {result['error']}\n"
            f"[bold]Confidence Threshold:[/bold] {result.get('confidence_threshold', 'N/A')}\n"
            f"[bold]Best Score:[/bold] {top_intents[0][1]:.3f} (from {top_intents[0][0].intent})\n\n"
            f"[yellow]💡 Tip:[/yellow] Lower the threshold with --threshold {top_intents[0][1]:.1f} to match the best result",
            title="No Match"
        )
    
    console.print(panel)


def get_variable_prompt(var_name: str, var_config: Dict[str, Any]) -> str:
    """Create user-friendly prompt for missing variable."""
    prompts = {
        'issue_key': "Which issue key? (e.g., PROJ-123)",
        'project': "Which project? (e.g., PROJ, ABC, MYPROJECT)",
        'status': "Target status? (e.g., In Progress, Done, QA)", 
        'assignee': "Assign to whom? (username or email)",
        'summary': "What's the summary/title?",
        'description': "Description (optional):",
        'priority': "Priority level? (High, Medium, Low)",
        'email': "Email address?",
        'amount': "Amount/value? (e.g., $50000, 1000)",
        'dealname': "Deal name?",
        'subject': "Subject/title?",
        'content': "Content/details:",
        'comment': "Comment text:",
        'label': "Label/tag name?",
        'sprint_name': "Sprint name? (e.g., Sprint 1, Alpha)",
        'firstname': "First name?",
        'lastname': "Last name?",
        'company': "Company name?",
        'jobtitle': "Job title?",
    }
    
    return prompts.get(var_name, f"Please provide {var_name}:")


def get_variable_examples(var_name: str) -> List[str]:
    """Get examples for a variable type."""
    examples = {
        'issue_key': ["PROJ-123", "ABC-456", "TASK-789"],
        'project': ["PROJ", "ABC", "MYPROJECT"],  
        'status': ["In Progress", "Done", "QA", "Review"],
        'email': ["john@company.com", "user@domain.org"],
        'amount': ["$50000", "1000", "$2,500.00"],
        'priority': ["High", "Medium", "Low"],
        'dealname': ["Enterprise License", "Q1 Deal"],
        'subject': ["Bug in login", "Feature request"],
        'assignee': ["john.doe", "jane@company.com"],
        'sprint_name': ["Sprint 1", "Alpha", "Q1 Sprint"]
    }
    
    return examples.get(var_name, [])


def interactive_mode(args):
    """Start interactive testing mode."""
    console = Console()
    console.print("[bold blue]🤖 Interactive Intent Matcher[/bold blue]")
    console.print("Type 'quit' to exit, 'stats' for statistics, 'help' for commands\n")
    
    # Load matcher
    if Path(args.index_dir).exists():
        console.print("[yellow]Loading existing index...[/yellow]")
        matcher = IntentMatcher(
            jira_config_path=args.jira_config,
            hubspot_config_path=args.hubspot_config,
            confidence_threshold=args.threshold
        )
        matcher.load_index(args.index_dir)
    else:
        console.print("[yellow]Building new index...[/yellow]")
        matcher = IntentMatcher(
            jira_config_path=args.jira_config,
            hubspot_config_path=args.hubspot_config,
            confidence_threshold=args.threshold,
            model_name=args.model
        )
    
    console.print("[green]Ready! Enter your queries:[/green]\n")
    
    while True:
        try:
            query = Prompt.ask("[bold cyan]Query")
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                stats = matcher.get_stats()
                rprint(json.dumps(stats, indent=2))
                continue
            elif query.lower() == 'help':
                console.print("""
[bold]Available commands:[/bold]
- quit: Exit interactive mode
- stats: Show system statistics  
- help: Show this help message
- Any other text: Process as intent query
                """)
                continue
            
            # Process query
            result = matcher.process_query(query)
            
            # Show results
            if result['success']:
                console.print(f"[green]✅ {result['intent']}[/green] ({result['platform']}) - {result['confidence']:.3f}")
                console.print(f"Variables: {result['variables']}")
                if Confirm.ask("Show full tool plan?", default=False):
                    rprint(json.dumps(result['tool_plan'], indent=2))
            else:
                console.print(f"[red]❌ {result['error']}[/red]")
                if 'suggestions' in result and result['suggestions']:
                    console.print("\n[yellow]Suggestions:[/yellow]")
                    for suggestion in result['suggestions'][:3]:
                        console.print(f"- {suggestion['intent']} ({suggestion['platform']}) - {suggestion['confidence']:.3f}")
            
            console.print()  # Empty line
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG-based Intent Matching System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py build                     # Build index with default settings
  python cli.py test "Create a bug"       # Test a single query
  python cli.py interactive               # Start interactive mode
  python cli.py build --threshold 0.85    # Build with custom threshold
        """
    )
    
    # Global arguments
    parser.add_argument('--jira-config', default='jira-intent-config.yaml',
                       help='Path to Jira configuration file')
    parser.add_argument('--hubspot-config', default='hubspot-intent-config.yaml',
                       help='Path to HubSpot configuration file')
    parser.add_argument('--threshold', type=float, 
                       default=float(os.getenv('CONFIDENCE_THRESHOLD', '0.85')),
                       help=f'Confidence threshold for intent matching (default: {os.getenv("CONFIDENCE_THRESHOLD", "0.85")})')
    parser.add_argument('--model', 
                       default=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                       help='Sentence transformer model name')
    parser.add_argument('--index-dir', 
                       default=os.getenv('INDEX_DIR', './models'),
                       help='Directory to save/load index files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build embedding index')
    build_parser.add_argument('--save', action='store_true', default=True,
                             help='Save index to disk (default: True)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a single query')
    test_parser.add_argument('query', help='Query string to test')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'build':
        build_index_command(args)
    elif args.command == 'test':
        test_query_command(args)
    elif args.command == 'interactive':
        interactive_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
