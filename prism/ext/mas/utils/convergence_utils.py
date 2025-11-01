import os
import ast
import numpy as np

from utils.common import read_json_file

from prism.embedding import EMBEDDING
from prism.ext.mas.utils.workflow_parser import WorkflowParser

class ConvergenceUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.embedding = EMBEDDING()
        self.threshold = 0.7
        
        self.critical_model_types = [
            "token-classification", "text-classification", "zero-shot-classification",
            "translation", "summarization", "question-answering", "text-generation",
            "sentence-similarity", "tabular-classification", "tabular-regression",
            "object-detection", "image-classification", "image-to-text",
            "automatic-speech-recognition", "audio-classification", "video-classification"
        ]
        
        # Weight for model type differences (higher = more strict)
        self.model_type_weight = 0.8
        # Weight for general semantic similarity
        self.semantic_weight = 0.2
        
        
    def check_convergence(self, directory: str, rounds: int) -> bool:
        experiences = []
        for i in range(1, rounds + 1):
            experience_file_path = os.path.join(directory, f"round_{i}", "experience.json")
            experiences.append(read_json_file(experience_file_path, encoding="utf-8"))
        if any(int(experience["score"]) == 1 for experience in experiences):
            return True
        return self._check_statistical_convergence(experiences)
    
    def _check_statistical_convergence(self, experiences, top_k=3, z=0.5, consecutive_rounds=3):
        """
        Check for statistical convergence using top-k analysis
        
        Args:
            experiences: List of experience dictionaries
            top_k: Number of top rounds to consider
            z: Z-score for confidence interval (0.5 = moderate, 1.0 = strict)
            consecutive_rounds: Number of consecutive stable rounds required
            
        Returns:
            bool: True if converged, False otherwise
        """
        scores = [float(exp["score"]) for exp in experiences]
        
        # Need at least top_k + 1 rounds for meaningful analysis
        if len(scores) < top_k + 1:
            return False
        
        # Estimate standard deviations (proportional to uncertainty)
        # Higher score = lower uncertainty
        estimated_stds = [max(0.01, (1 - score) * 0.2) for score in scores]
        
        convergence_count = 0
        previous_y = None
        sigma_y_previous = None
        
        for i in range(len(scores)):
            # Get top_k indices from current round and all previous rounds
            available_scores = scores[:i+1]
            available_stds = estimated_stds[:i+1]
            
            # Select top_k rounds by descending score
            top_k_indices = np.argsort(available_scores)[::-1][:top_k]
            top_k_scores = [available_scores[j] for j in top_k_indices]
            top_k_stds = [available_stds[j] for j in top_k_indices]
            
            # Calculate mean and standard error of top_k scores
            y_current = np.mean(top_k_scores)
            sigma_y_current = np.sqrt(sum(s**2 for s in top_k_stds) / (top_k**2))
            
            # Check convergence condition
            if previous_y is not None and sigma_y_previous is not None:
                delta_y = y_current - previous_y
                sigma_delta_y = np.sqrt(sigma_y_current**2 + sigma_y_previous**2)
                
                # Statistical stability test: change must be within confidence interval
                if abs(delta_y) <= z * sigma_delta_y:
                    convergence_count += 1
                    if convergence_count >= consecutive_rounds:
                        print("Converged")
                        return True
                else:
                    # Reset counter if change is too large
                    convergence_count = 0
            
            previous_y = y_current
            sigma_y_previous = sigma_y_current
        
        return False
    
    async def check_modification(self, directory: str, rounds: int, round_number: int) -> bool:
        experiences = []
        current_workflow_file_path = os.path.join(directory, f"round_{round_number}", "workflow.py")
        current_workflow = open(current_workflow_file_path, "r", encoding="utf-8").read()
        
        for i in range(1, rounds):
            experience_file_path = os.path.join(directory, f"round_{i}", "experience.json")
            experiences.append(read_json_file(experience_file_path, encoding="utf-8"))

        similar_workflows = []

        for experience in experiences:
            if await self.compare_workflows(current_workflow, experience["workflow"]):
                similar_workflows.append(experience)
                
    
    async def compare_workflows(self, workflow1: str, workflow2: str) -> bool:
        structure_similarity = await self.compare_workflow_structures(workflow1, workflow2)
        if not structure_similarity:
            return False
        
        instruction_similarity = await self.compare_workflow_instructions(workflow1, workflow2)
        return instruction_similarity
    
    async def compare_workflow_structures(self, workflow1: str, workflow2: str) -> bool:
        parser1 = WorkflowParser()
        parser2 = WorkflowParser()
        
        tree1 = ast.parse(workflow1)
        tree2 = ast.parse(workflow2)
        
        parser1.visit(tree1)
        parser2.visit(tree2)
        
        structure1 = parser1.get_workflow_structure()
        structure2 = parser2.get_workflow_structure()
        
        comparison = {
            'agents_match': set(structure1['agents']) == set(structure2['agents']),
            'call_order_match': structure1['call_order'] == structure2['call_order'],
            'dependency_match': parser1.get_dependency_graph() == parser2.get_dependency_graph(),
            'structure1': structure1,
            'structure2': structure2,
            'dependency_graph1': parser1.get_dependency_graph(),
            'dependency_graph2': parser2.get_dependency_graph()
        }
        
        matches = [
            comparison["agents_match"],
            comparison["call_order_match"],
            comparison["dependency_match"]
        ]
        comparison["similarity_score"] = sum(matches) / len(matches)
        comparison["is_structural_match"] = comparison["similarity_score"] >= 0.8
        print("Structure Score: ", comparison["similarity_score"])
        return comparison["is_structural_match"]


    def _extract_model_types(self, instruction: str) -> set:
        """Extract model types from instruction text"""
        found_types = set()
        instruction_lower = instruction.lower()
        
        for model_type in self.critical_model_types:
            if model_type in instruction_lower:
                found_types.add(model_type)
        
        return found_types
    
    def _compute_model_type_similarity(self, instruction1: str, instruction2: str) -> float:
        """Compute similarity based on model types used"""
        types1 = self._extract_model_types(instruction1)
        types2 = self._extract_model_types(instruction2)
        
        # If no model types found in either, treat as neutral (don't penalize)
        if not types1 and not types2:
            return 1.0
        
        # If one has model types and the other doesn't, they're different
        if (types1 and not types2) or (types2 and not types1):
            return 0.0
        
        # If both have model types, check overlap
        if types1 and types2:
            intersection = types1.intersection(types2)
            union = types1.union(types2)
            
            # Jaccard similarity
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
            return jaccard_similarity
        
        return 1.0

    async def compare_workflow_instructions(self, workflow1: str, workflow2: str) -> bool:
        """
        Enhanced version with model type weighting and early stopping
        """
        try:
            # Parse workflows
            parser1 = WorkflowParser()
            parser2 = WorkflowParser()
            
            tree1 = ast.parse(workflow1)
            tree2 = ast.parse(workflow2)
            
            parser1.visit(tree1)
            parser2.visit(tree2)
            
            instructions1 = list(parser1.instructions.values())
            instructions2 = list(parser2.instructions.values())
            
            # Filter out empty instructions
            instructions1 = [inst for inst in instructions1 if inst.strip()]
            instructions2 = [inst for inst in instructions2 if inst.strip()]
            
            if len(instructions1) != len(instructions2):
                return False
                
            if not instructions1:  # Both empty
                return True
            
            # Early stopping approach with weighted similarity
            min_acceptable_similarity = max(0.5, self.threshold - 0.2)
            similarities = []
            
            for i, (inst1, inst2) in enumerate(zip(instructions1, instructions2)):
                # Compute both types of similarity
                semantic_sim = await self._compute_instruction_similarity(inst1, inst2)
                model_type_sim = self._compute_model_type_similarity(inst1, inst2)
                
                # Weighted combination
                weighted_similarity = (
                    self.semantic_weight * semantic_sim + 
                    self.model_type_weight * model_type_sim
                )
                
                similarities.append(weighted_similarity)
                
                print(f"Step {i+1}:")
                print(f"  Semantic similarity: {semantic_sim:.3f}")
                print(f"  Model type similarity: {model_type_sim:.3f}")
                print(f"  Weighted similarity: {weighted_similarity:.3f}")
                
                # Early stopping if this instruction is too different
                if weighted_similarity < min_acceptable_similarity:
                    print(f"Early stop: Step {i+1} weighted similarity {weighted_similarity:.3f} < {min_acceptable_similarity:.3f}")
                    return False
            
            # Check overall average
            avg_similarity = sum(similarities) / len(similarities)
            print(f"Average weighted similarity: {avg_similarity:.3f}")
            
            return avg_similarity >= self.threshold
            
        except Exception as e:
            print(f"Error comparing workflow instructions: {e}")
            return False
            
    
    async def _compute_instruction_similarity(self, instruction1: str, instruction2: str) -> float:
        """Compute similarity between two instructions using embeddings"""
        try:
            # Clean instructions
            inst1_clean = instruction1.strip().replace('\n', ' ')
            inst2_clean = instruction2.strip().replace('\n', ' ')
            
            if not inst1_clean or not inst2_clean:
                return 0.0
            
            if inst1_clean.lower() == inst2_clean.lower():
                return 1.0
            
            # Get embeddings
            embeddings = await self.embedding.encode([inst1_clean, inst2_clean])
            
            if len(embeddings) != 2:
                return 0.0
            
            emb1 = embeddings[0]
            emb2 = embeddings[1]
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            
            # Convert from [-1, 1] to [0, 1] range
            normalized_similarity = (cosine_similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            print(f"Error computing instruction similarity: {e}")
            return 0.0
    
    
async def test():
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
        glue_result = await self.glue_coder.run(instruction="Use glue-coder to classify the input image into one of the 1000 ImageNet categories", experience=[])
        return await self.synthesizer.run(experience=[classification_result, glue_result])"""
        
    workflow2 = """import pandas as pd

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
        result = await self.ml_coder.run(instruction="Use object-detection to classify the input image into one of the 1000 ImageNet categories", experience=[])
        glue_result = await self.glue_coder.run(instruction="Use glue-coder to classify the input image into one of the 1000 ImageNet categories", experience=[])
        return await self.synthesizer.run(experience=[result, glue_result])"""
        
    convergence_utils = ConvergenceUtils(root_path="prism/ext/mas")
    print(await convergence_utils.compare_workflows(workflow1, workflow2))
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test())