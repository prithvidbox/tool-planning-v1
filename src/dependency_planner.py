"""
Dependency-aware tool planning system for ensuring proper tool execution order.
"""

from typing import Dict, List, Any, Set, Tuple, Optional
from loguru import logger
from .config_parser import IntentConfig


class ToolDependency:
    """Represents a tool and its variable dependencies."""
    
    def __init__(self, tool_step: Dict[str, Any], step_index: int):
        self.tool_name = tool_step.get('tool')
        self.params = tool_step.get('params', {})
        self.post_process = tool_step.get('post_process')
        self.note = tool_step.get('note')
        self.step_index = step_index
        self.original_step = tool_step
        
        # Variables this tool needs
        self.required_vars = set()
        for param, value in self.params.items():
            if isinstance(value, str) and value.startswith('$'):
                # Handle default value syntax: $variable || "default"
                if '||' in value:
                    var_part = value.split('||')[0].strip()
                    if var_part.startswith('$'):
                        var_name = var_part[1:]  # Remove $ prefix
                        self.required_vars.add(var_name)
                else:
                    var_name = value[1:]  # Remove $ prefix
                    self.required_vars.add(var_name)
        
        # Variables this tool provides (through post-processing)
        self.provided_vars = set()
        if self.post_process:
            self.provided_vars.update(self._get_provided_variables())
    
    def _get_provided_variables(self) -> Set[str]:
        """Determine what variables this tool provides based on post-processing."""
        # Map post-processors to the variables they provide
        post_processor_outputs = {
            'find_transition_id': {'transition_id'},
            'extract_end_date': {'end_date'},
            'extract_attachments': {'attachments'},
            'count_results': {'count', 'total'},
            'build_associations': {'association_ids'},
            'generate_burndown_chart': {'burndown_data'},
            'calculate_average_resolution_time': {'average_time'},
            'generate_closure_report': {'closure_report'},
            'calculate_velocity_last_3_sprints': {'velocity_data'},
            'format_notification': {'notification_text'},
            'schedule_reminder': {'reminder_id'},
            'setup_alert_subscription': {'subscription_id'},
            'format_qa_notification': {'qa_notification'},
            'generate_daily_summary': {'daily_summary'},
            'format_and_share_external': {'share_result'},
            'group_by_stage': {'stage_groups'},
            'group_by_lifecycle_stage': {'lifecycle_groups'},
            'calculate_revenue_forecast': {'revenue_forecast'},
            'generate_activity_summary': {'activity_summary'},
            'format_pipelines': {'pipeline_data'},
            'bulk_import_contacts': {'import_results'},
            'bulk_update_lifecycle_stage': {'update_results'},
        }
        
        if self.post_process in post_processor_outputs:
            return post_processor_outputs[self.post_process]
        
        return set()
    
    def can_execute_after(self, available_vars: Set[str]) -> bool:
        """Check if this tool can execute given available variables."""
        return self.required_vars.issubset(available_vars)
    
    def __repr__(self):
        return f"ToolDependency(tool={self.tool_name}, needs={self.required_vars}, provides={self.provided_vars})"


class DependencyPlanner:
    """Plans tool execution order based on variable dependencies."""
    
    @staticmethod
    def analyze_dependencies(tool_plan: List[Dict[str, Any]]) -> List[ToolDependency]:
        """Analyze dependencies for all tools in a plan."""
        dependencies = []
        for i, tool_step in enumerate(tool_plan):
            dep = ToolDependency(tool_step, i)
            dependencies.append(dep)
            logger.debug(f"Tool {dep.tool_name}: needs {dep.required_vars}, provides {dep.provided_vars}")
        return dependencies
    
    @staticmethod
    def validate_dependencies(dependencies: List[ToolDependency], 
                             available_vars: Set[str], 
                             intent_config = None) -> Tuple[bool, List[str], List[str]]:
        """Validate that all dependencies can be satisfied."""
        current_vars = available_vars.copy()
        missing_required = []
        missing_optional = []
        
        # Get required vs optional variables from intent config
        required_vars = set()
        optional_vars = set()
        
        if intent_config and intent_config.variables:
            for var_def in intent_config.variables:
                var_name = var_def.get('name')
                is_required = var_def.get('required', False)
                if is_required:
                    required_vars.add(var_name)
                else:
                    optional_vars.add(var_name)
        
        for dep in dependencies:
            # Check which variables this tool actually needs
            tool_required = set()
            tool_optional = set()
            
            for param, value in dep.params.items():
                if isinstance(value, str) and value.startswith('$'):
                    if '||' in value:
                        # Has default value syntax, skip
                        continue
                    else:
                        var_name = value[1:]
                        # Classify based on intent config
                        if var_name in required_vars:
                            tool_required.add(var_name)
                        elif var_name in optional_vars:
                            tool_optional.add(var_name)
                        else:
                            # Default to required if not specified in config
                            tool_required.add(var_name)
            
            # Check missing variables
            missing_req = tool_required - current_vars
            missing_opt = tool_optional - current_vars
            
            if missing_req:
                missing_required.extend(missing_req)
                logger.warning(f"Tool {dep.tool_name} missing required variables: {missing_req}")
            
            if missing_opt:
                missing_optional.extend(missing_opt)
                logger.info(f"Tool {dep.tool_name} missing optional variables: {missing_opt}")
            
            # Tool can execute, add its provided variables
            current_vars.update(dep.provided_vars)
                
        return len(missing_required) == 0, list(set(missing_required)), list(set(missing_optional))
    
    @staticmethod
    def reorder_tool_plan(dependencies: List[ToolDependency],
                         available_vars: Set[str]) -> List[Dict[str, Any]]:
        """Reorder tools based on dependency requirements."""
        ordered_tools = []
        remaining_deps = dependencies.copy()
        current_vars = available_vars.copy()
        
        # Keep trying to find executable tools until none remain
        while remaining_deps:
            executed_in_round = False
            
            # Find tools that can execute with current variables (including those with defaults)
            executable_now = []
            for i, dep in enumerate(remaining_deps):
                # Check if tool can execute (considering default values)
                truly_required = set()
                for param, value in dep.params.items():
                    if isinstance(value, str) and value.startswith('$'):
                        if '||' not in value:  # Only count variables without defaults
                            var_name = value[1:]
                            truly_required.add(var_name)
                
                if truly_required.issubset(current_vars):
                    executable_now.append((i, dep))
            
            if not executable_now:
                # No tools can execute with current variables - try to execute remaining with defaults
                logger.warning(f"Some tools may use default values: {[dep.tool_name for dep in remaining_deps]}")
                # Execute all remaining tools (they have defaults)
                for i, dep in enumerate(remaining_deps):
                    executable_now.append((i, dep))
            
            # Sort by original step index to maintain relative order when possible
            executable_now.sort(key=lambda x: x[1].step_index)
            
            # Execute tools that can run now
            for index, dep in reversed(executable_now):  # Reverse to maintain indices
                ordered_tools.append(dep.original_step)
                current_vars.update(dep.provided_vars)
                remaining_deps.pop(index)
                executed_in_round = True
                logger.debug(f"Scheduled {dep.tool_name}, now have vars: {current_vars}")
            
            if not executed_in_round:
                break
                
        return ordered_tools
    
    @classmethod
    def plan_tool_execution(cls, intent_config: IntentConfig, 
                           extracted_vars: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan tool execution with proper dependency ordering.
        
        Args:
            intent_config: Intent configuration with tool plan
            extracted_vars: Variables extracted from user query
            
        Returns:
            Dict with success status and ordered tool plan or error details
        """
        try:
            if not intent_config.tool_plan:
                return {
                    'success': False,
                    'error': 'No tool plan defined for this intent'
                }
            
            # Analyze dependencies
            dependencies = cls.analyze_dependencies(intent_config.tool_plan)
            
            # Get available variables from extraction
            available_vars = set(extracted_vars.keys())
            logger.info(f"Available variables from query: {available_vars}")
            
            # Validate that all dependencies can be satisfied
            is_valid, missing_required, missing_optional = cls.validate_dependencies(
                dependencies, available_vars, intent_config
            )
            
            if not is_valid:
                return {
                    'success': False,
                    'error': f'Cannot satisfy dependencies. Missing required variables: {", ".join(missing_required)}',
                    'missing_variables': missing_required,
                    'missing_optional_variables': missing_optional,
                    'available_variables': list(available_vars)
                }
            
            # Reorder tools based on dependencies
            ordered_plan = cls.reorder_tool_plan(dependencies, available_vars)
            
            # Replace variables in the ordered plan
            final_plan = []
            for tool_step in ordered_plan:
                tool_name = tool_step.get('tool')
                raw_params = tool_step.get('params', {})
                
                # Replace variables in parameters
                params = cls._replace_variables(raw_params, extracted_vars)
                
                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}
                
                final_plan.append({
                    'tool': tool_name,
                    'params': params,
                    'post_process': tool_step.get('post_process'),
                    'note': tool_step.get('note')
                })
            
            return {
                'success': True,
                'tool_plan': final_plan,
                'execution_order': [step['tool'] for step in final_plan],
                'dependency_analysis': {
                    'total_tools': len(dependencies),
                    'reordered': len(ordered_plan) != len(intent_config.tool_plan) or 
                               [step['tool'] for step in ordered_plan] != [step['tool'] for step in intent_config.tool_plan],
                    'available_vars': list(available_vars),
                    'tools_with_dependencies': [dep.tool_name for dep in dependencies if dep.required_vars]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in dependency planning: {e}")
            return {
                'success': False,
                'error': f'Dependency planning failed: {str(e)}'
            }
    
    @staticmethod
    def _replace_variables(params: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
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


class SmartToolPlanner:
    """Enhanced tool planner with dependency awareness and intelligent variable gathering."""
    
    @staticmethod
    def create_variable_gathering_plan(missing_vars: List[str], platform: str) -> List[Dict[str, Any]]:
        """
        Create additional tool steps to gather missing variables.
        
        Args:
            missing_vars: Variables that need to be gathered
            platform: Target platform (jira/hubspot)
            
        Returns:
            List of tool steps to gather the missing variables
        """
        gathering_plan = []
        
        if platform == 'jira':
            if 'project_info' in missing_vars:
                gathering_plan.append({
                    'tool': 'list_projects',
                    'params': {},
                    'post_process': 'extract_default_project',
                    'note': 'Get available projects for user selection'
                })
            
            if 'available_transitions' in missing_vars:
                gathering_plan.append({
                    'tool': 'get_issue_transitions', 
                    'params': {'issueKey': '$issue_key'},
                    'post_process': 'list_available_statuses',
                    'note': 'Get possible status transitions'
                })
                
        elif platform == 'hubspot':
            if 'contact_info' in missing_vars:
                gathering_plan.append({
                    'tool': 'search_contacts',
                    'params': {'limit': 10},
                    'post_process': 'suggest_contacts',
                    'note': 'Get recent contacts for selection'
                })
                
        return gathering_plan
    
    @staticmethod
    def create_smart_fallback_plan(intent_config: IntentConfig, 
                                  partial_vars: Dict[str, Any],
                                  missing_vars: List[str]) -> Dict[str, Any]:
        """
        Create a fallback plan that can gather missing variables interactively.
        
        This allows the system to partially execute a plan and ask for missing info.
        """
        available_vars = set(partial_vars.keys())
        
        # Find tools that can execute with current variables
        executable_tools = []
        blocked_tools = []
        
        for tool_step in intent_config.tool_plan:
            dep = ToolDependency(tool_step, 0)
            if dep.can_execute_after(available_vars):
                executable_tools.append(tool_step)
                available_vars.update(dep.provided_vars)
            else:
                blocked_tools.append({
                    'tool': dep.tool_name,
                    'missing': list(dep.required_vars - available_vars)
                })
        
        return {
            'partial_execution_possible': len(executable_tools) > 0,
            'executable_tools': executable_tools,
            'blocked_tools': blocked_tools,
            'missing_for_completion': missing_vars,
            'suggestion': f"I can execute {len(executable_tools)} of {len(intent_config.tool_plan)} tools. "
                         f"To complete the task, I need: {', '.join(missing_vars)}"
        }


# Example of enhanced intent configurations with dependency declarations
DEPENDENCY_EXAMPLES = {
    'jira_multi_step_examples': [
        {
            'intent': 'create_and_assign_bug',
            'description': 'Create a bug and assign it to someone',
            'tool_plan': [
                {
                    'tool': 'create_issue',
                    'params': {
                        'projectKey': '$project',
                        'summary': '$summary',
                        'issueType': 'Bug'
                    },
                    'post_process': 'extract_issue_key',
                    'provides': ['issue_key']  # This tool will provide issue_key
                },
                {
                    'tool': 'assign_issue', 
                    'params': {
                        'issueKey': '$issue_key',  # Depends on previous tool
                        'assignee': '$assignee'
                    },
                    'requires': ['issue_key']  # This tool requires issue_key
                }
            ]
        }
    ],
    
    'hubspot_multi_step_examples': [
        {
            'intent': 'create_contact_and_deal',
            'description': 'Create a contact and then create a deal for them',
            'tool_plan': [
                {
                    'tool': 'create_contact',
                    'params': {
                        'email': '$email',
                        'firstname': '$firstname',
                        'lastname': '$lastname'
                    },
                    'post_process': 'extract_contact_id',
                    'provides': ['contact_id']
                },
                {
                    'tool': 'create_deal',
                    'params': {
                        'dealname': '$deal_name',
                        'amount': '$amount'
                    },
                    'post_process': 'extract_deal_id', 
                    'provides': ['deal_id']
                },
                {
                    'tool': 'create_association',
                    'params': {
                        'fromObjectType': 'contacts',
                        'fromObjectId': '$contact_id',  # From step 1
                        'toObjectType': 'deals',
                        'toObjectId': '$deal_id',       # From step 2
                        'associationType': 'primary'
                    },
                    'requires': ['contact_id', 'deal_id']
                }
            ]
        }
    ]
}
