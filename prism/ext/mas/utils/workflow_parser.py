import ast

class WorkflowParser(ast.NodeVisitor):
    def __init__(self):
        self.current_agent = None
        self.call_order = []
        self.dependencies = {}
        self.agents = set()
        self.instructions = {}  # Store instructions for each agent call
        self.variable_to_agent = {}  # Map variable names to agents

    def visit_Assign(self, node):
        # Detect assignments like classification_result = await agent.run(...)
        if isinstance(node.value, ast.Await):
            call = node.value.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                # Extract agent name from self.agent_name (e.g., self.ml_coder)
                if isinstance(call.func.value, ast.Attribute):
                    agent_name = call.func.value.attr  # Get the attribute name (e.g., ml_coder)
                    variable_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                    
                    self.agents.add(agent_name)
                    self.current_agent = agent_name
                    self.call_order.append(agent_name)
                    
                    # Map variable to agent for dependency tracking
                    if variable_name:
                        self.variable_to_agent[variable_name] = agent_name
                    
                    # Extract instruction
                    instruction = ""
                    for kw in call.keywords:
                        if kw.arg == "instruction" and isinstance(kw.value, ast.Constant):
                            instruction = kw.value.value
                    
                    # Store instruction
                    call_id = f"{agent_name}_{len([a for a in self.call_order if a == agent_name])}"
                    self.instructions[call_id] = instruction
                    
                    # Check if experience is passed and extract dependencies
                    for kw in call.keywords:
                        if kw.arg == "experience" and isinstance(kw.value, ast.List):
                            dependencies = []
                            for elt in kw.value.elts:
                                if isinstance(elt, ast.Name):
                                    # Map variable back to agent
                                    dep_variable = elt.id
                                    dependencies.append(dep_variable)
                                    
                            if dependencies:
                                if agent_name not in self.dependencies:
                                    self.dependencies[agent_name] = []
                                # Avoid duplicate dependencies for same agent
                                unique_deps = list(set(dependencies) - set(self.dependencies[agent_name]))
                                self.dependencies[agent_name].extend(unique_deps)
        
        self.generic_visit(node)
    
    def visit_Return(self, node):
        # Handle return await self.synthesizer.run(experience=[...])
        if isinstance(node.value, ast.Await):
            call = node.value.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                if isinstance(call.func.value, ast.Attribute):
                    agent_name = call.func.value.attr
                    
                    self.agents.add(agent_name)
                    self.call_order.append(agent_name)
                    
                    # Extract dependencies from experience
                    for kw in call.keywords:
                        if kw.arg == "experience" and isinstance(kw.value, ast.List):
                            dependencies = []
                            for elt in kw.value.elts:
                                if isinstance(elt, ast.Name):
                                    dependencies.append(elt.id)
                            
                            if dependencies:
                                if agent_name not in self.dependencies:
                                    self.dependencies[agent_name] = []
                                self.dependencies[agent_name] = list(set(dependencies))
        
        self.generic_visit(node)
    
    def get_workflow_structure(self):
        """Get a structured representation of the workflow"""
        return {
            'agents': list(self.agents),
            'call_order': self.call_order,
            'dependencies': self.dependencies,
            'instructions': self.instructions,
            'variable_to_agent': self.variable_to_agent
        }
    
    def get_dependency_graph(self):
        """Create a dependency graph between agents"""
        agent_dependencies = {}
        
        for agent, var_deps in self.dependencies.items():
            agent_deps = []
            for var in var_deps:
                if var in self.variable_to_agent:
                    dep_agent = self.variable_to_agent[var]
                    if dep_agent != agent:  # Avoid self-dependency
                        agent_deps.append(dep_agent)
            
            if agent_deps:
                agent_dependencies[agent] = list(set(agent_deps))
        
        return agent_dependencies
    
    def reset(self):
        """Reset parser state for reuse"""
        self.current_agent = None
        self.call_order = []
        self.dependencies = {}
        self.agents = set()
        self.instructions = {}
        self.variable_to_agent = {}

    
    
        
if __name__ == "__main__":
    workflow_parser = WorkflowParser()
    workflow1 = """import pandas as pd

from prism.ext.mas.agent.glue_coder import GlueCoder
from prism.ext.mas.agent.ml_coder import MLCoder
from prism.ext.mas.agent.synthesizer import Synthesizer
from prism.schema.message import Message

class Workflow:
    def __init__(self, problem: str, data: pd.DataFrame):
        self.problem = problem
        self.glue_coder = GlueCoder(problem=self.problem, data=data)
        self.ml_coder = MLCoder(problem=self.problem, data=data)
        self.synthesizer = Synthesizer(problem=self.problem, data=data)
        
    async def __call__(self):
        # Current data type: IMAGE - using image-classification (CORRECT for image data)
        classification_result = await self.ml_coder.run(instruction="Use image-classification to classify the input image into one of the 1000 ImageNet categories", experience=[])
        classification_result_2 = await self.ml_coder.run(instruction="Use image-classification to classify the input image into one of the 1000 ImageNet categories", experience=[])
        return await self.synthesizer.run(experience=[classification_result])"""
    workflow_parser.visit(ast.parse(workflow1))
    print(workflow_parser.get_workflow_structure())