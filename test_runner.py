#!/usr/bin/env python3
"""
Automated test runner for 100 query benchmark with AI-generated variable filling.
"""

import subprocess
import json
import time
import re
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()

class QueryTestRunner:
    """Automated test runner for intent matching system."""
    
    def __init__(self):
        self.console = Console()
        self.results = []
        self.total_time = 0
        self.total_tokens = 0
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def parse_test_queries(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse the test queries file."""
        queries = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '|' in line:
                # Parse format: Query | Expected Intent | Expected Platform | Variables
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    # Extract query number and text
                    query_part = parts[0]
                    match = re.match(r'(\d+)\.\s*(.+)', query_part)
                    if match:
                        query_num = int(match.group(1))
                        query_text = match.group(2)
                        
                        queries.append({
                            'id': query_num,
                            'query': query_text,
                            'expected_intent': parts[1],
                            'expected_platform': parts[2],
                            'expected_vars': parts[3] if parts[3] != '(none)' else ''
                        })
        
        return queries
    
    def generate_missing_variables(self, missing_vars: List[str], query: str) -> Dict[str, str]:
        """Use OpenAI to generate realistic values for missing variables."""
        if not missing_vars:
            return {}
            
        try:
            prompt = f"""Generate realistic values for these missing variables based on the query: "{query}"

Variables needed: {', '.join(missing_vars)}

Rules:
- issue_key: Use format like PROJ-123, TASK-456, BUG-789
- project: Use format like PROJ, TASK, BUG, ABC
- email: Use realistic emails like john@example.com
- amount: Use realistic amounts like $25000, $50000
- assignee: Use usernames like john.doe, jane.smith
- priority: Use High, Medium, or Low
- status: Use In Progress, Done, QA, Review
- firstname/lastname: Use common names
- company: Use realistic company names
- summary/subject: Create relevant titles
- description/content: Create brief descriptions

Return as JSON:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate realistic test data as JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except:
                pass
                
        except Exception as e:
            self.console.print(f"[red]AI generation failed: {e}[/red]")
        
        # Fallback to defaults
        defaults = {
            'issue_key': 'PROJ-123',
            'project': 'PROJ',
            'email': 'test@example.com',
            'amount': '$50000',
            'assignee': 'john.doe',
            'priority': 'Medium',
            'status': 'In Progress',
            'firstname': 'John',
            'lastname': 'Smith',
            'company': 'Example Corp',
            'summary': 'Test task',
            'subject': 'Test subject',
            'description': 'Test description',
            'content': 'Test content'
        }
        
        return {var: defaults.get(var, 'test_value') for var in missing_vars}
    
    def run_cli_query(self, query: str, threshold: float = 0.35) -> Dict[str, Any]:
        """Run a single query through the CLI and parse results."""
        try:
            # Run CLI command
            cmd = f'python3 cli.py --threshold {threshold} test "{query}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': result.stderr,
                    'raw_output': result.stdout
                }
            
            output = result.stdout
            
            # Parse the output for key metrics
            parsed_result = {
                'success': True,
                'raw_output': output,
                'confidence': self._extract_confidence(output),
                'intent': self._extract_intent(output),
                'platform': self._extract_platform(output),
                'variables': self._extract_variables(output),
                'tools': self._extract_tools(output),
                'performance': self._extract_performance(output)
            }
            
            return parsed_result
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout', 'timeout': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_confidence(self, output: str) -> Optional[float]:
        """Extract confidence score from CLI output."""
        match = re.search(r'Confidence: ([\d.]+)', output)
        return float(match.group(1)) if match else None
    
    def _extract_intent(self, output: str) -> Optional[str]:
        """Extract matched intent from CLI output."""
        match = re.search(r'Intent: ([^\n]+)', output)
        return match.group(1).strip() if match else None
    
    def _extract_platform(self, output: str) -> Optional[str]:
        """Extract platform from CLI output."""
        match = re.search(r'Platform: ([^\n]+)', output)
        return match.group(1).strip() if match else None
    
    def _extract_variables(self, output: str) -> Dict[str, Any]:
        """Extract variables from CLI output."""
        try:
            # Look for Variables Extracted section
            match = re.search(r'Variables Extracted:\s*\n\s*(\{[^}]*\})', output, re.DOTALL)
            if match:
                json_str = match.group(1).replace('\n', '').replace('  ', ' ')
                return json.loads(json_str)
        except:
            pass
        return {}
    
    def _extract_tools(self, output: str) -> List[str]:
        """Extract execution order from CLI output."""
        match = re.search(r'Execution Order: ([^\n]+)', output)
        if match:
            tools_str = match.group(1).strip()
            return [t.strip() for t in tools_str.split('→')]
        return []
    
    def _extract_performance(self, output: str) -> Dict[str, Any]:
        """Extract performance metrics from CLI output."""
        perf = {}
        
        # Extract timing
        time_match = re.search(r'Variable Extraction: ([\d.]+)s', output)
        if time_match:
            perf['extraction_time'] = float(time_match.group(1))
            
        # Extract tokens
        token_match = re.search(r'Total: [\d.]+s \((\d+) tokens\)', output)
        if token_match:
            perf['total_tokens'] = int(token_match.group(1))
            
        # Extract cache hit
        cache_match = re.search(r'Cache Hit: (Yes|No)', output)
        if cache_match:
            perf['cache_hit'] = cache_match.group(1) == 'Yes'
            
        return perf
    
    def run_all_tests(self, test_file: str = 'test_queries_100.txt'):
        """Run all test queries and collect results."""
        queries = self.parse_test_queries(test_file)
        
        self.console.print(f"[bold blue]🧪 Running {len(queries)} test queries...[/bold blue]")
        
        with Progress() as progress:
            task = progress.add_task("Processing queries...", total=len(queries))
            
            for query_data in queries:
                start_time = time.time()
                
                # Run the query
                result = self.run_cli_query(query_data['query'])
                
                # Calculate timing
                end_time = time.time()
                query_time = end_time - start_time
                
                # Store result
                test_result = {
                    **query_data,
                    **result,
                    'query_time': query_time,
                    'timestamp': time.time()
                }
                
                self.results.append(test_result)
                
                # Track totals
                if result.get('success') and result.get('performance'):
                    perf = result['performance']
                    if 'total_tokens' in perf:
                        self.total_tokens += perf['total_tokens']
                    if 'extraction_time' in perf:
                        self.total_time += perf['extraction_time']
                
                # Update progress
                progress.update(task, advance=1)
                
                # Brief pause to be respectful to OpenAI API
                time.sleep(0.1)
        
        self.console.print(f"[green]✅ Completed all {len(queries)} tests[/green]")
        
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate summary."""
        if not self.results:
            return {}
            
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]
        
        # Platform breakdown
        platforms = {}
        for result in successful:
            platform = result.get('platform', 'unknown')
            platforms[platform] = platforms.get(platform, 0) + 1
        
        # Confidence analysis
        confidences = [r.get('confidence', 0) for r in successful if r.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Performance analysis
        times = [r.get('performance', {}).get('extraction_time', 0) for r in successful]
        avg_time = sum(times) / len(times) if times else 0
        
        tokens = [r.get('performance', {}).get('total_tokens', 0) for r in successful]
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0
        
        # Intent accuracy
        correct_intents = 0
        for result in successful:
            if result.get('intent') == result.get('expected_intent'):
                correct_intents += 1
        
        intent_accuracy = correct_intents / len(successful) if successful else 0
        
        return {
            'total_queries': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.results),
            'platform_breakdown': platforms,
            'avg_confidence': avg_confidence,
            'avg_extraction_time': avg_time,
            'avg_tokens_per_query': avg_tokens,
            'total_tokens_used': self.total_tokens,
            'intent_accuracy': intent_accuracy,
            'total_test_time': sum(r.get('query_time', 0) for r in self.results)
        }
    
    def save_results(self, output_file: str = 'test_results_100.json'):
        """Save detailed results to file."""
        analysis = self.analyze_results()
        
        output_data = {
            'summary': analysis,
            'detailed_results': self.results,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_notes': 'OpenAI-only system with ultra-concise prompts'
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        self.console.print(f"[green]📊 Results saved to {output_file}[/green]")
        
    def print_summary(self):
        """Print summary of test results."""
        analysis = self.analyze_results()
        
        table = Table(title="🧪 Test Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Queries", str(analysis['total_queries']))
        table.add_row("Success Rate", f"{analysis['success_rate']:.1%}")
        table.add_row("Intent Accuracy", f"{analysis['intent_accuracy']:.1%}")
        table.add_row("Avg Confidence", f"{analysis['avg_confidence']:.3f}")
        table.add_row("Avg Time per Query", f"{analysis['avg_extraction_time']:.3f}s")
        table.add_row("Avg Tokens per Query", f"{analysis['avg_tokens_per_query']:.1f}")
        table.add_row("Total Tokens Used", str(analysis['total_tokens_used']))
        table.add_row("Total Test Time", f"{analysis['total_test_time']:.1f}s")
        table.add_row("Platform Breakdown", str(analysis['platform_breakdown']))
        
        self.console.print(table)


def main():
    """Main execution."""
    console = Console()
    
    console.print("[bold blue]🚀 Starting 100 Query Benchmark Test[/bold blue]")
    console.print("[yellow]This will test all queries with OpenAI and generate missing variables automatically[/yellow]\n")
    
    runner = QueryTestRunner()
    
    # Run all tests
    runner.run_all_tests()
    
    # Print summary
    runner.print_summary()
    
    # Save detailed results
    runner.save_results()
    
    console.print("\n[bold green]🎉 Benchmark completed![/bold green]")


if __name__ == '__main__':
    main()
