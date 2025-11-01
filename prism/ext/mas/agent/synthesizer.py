import sys
import json
import traceback
import importlib.util
import pandas as pd
from pydantic import model_validator
from typing_extensions import Self

from utils.logs import logger
from utils.common import describe_dict

from prism.schema.message import Message

from prism.ext.mas.agent.coder import Coder
from prism.ext.mas.actions.synthesize_solution import SynthesizeSolution

def _traceback_str(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

class Synthesizer(Coder):
    name: str = "Synthesize Coder"
    profile: str = "A professional synthesizer"
    
    @model_validator(mode="after")
    def set_up(self) -> Self:
        self.set_actions([SynthesizeSolution])
        self._set_state(0)
        return self
    
    def format_experience(self, experience: list[Message]) -> list[Message]:
        """Format experience recursively from oldest to newest, avoiding duplicates"""
        
        def collect_all_experiences(experiences: list[Message]) -> list[Message]:
            """Recursively collect all experiences, starting from the oldest"""
            all_old_experiences = []
            current_experiences = []
            
            for exp in experiences:
                # First, collect all previous experiences (older ones) recursively
                previous_experiences = exp.metadata.get('previous_experience')
                if previous_experiences:
                    # Recursively get older experiences first
                    older_experiences = collect_all_experiences(previous_experiences)
                    all_old_experiences.extend(older_experiences)
                
                # Add current experience to the current list
                current_experiences.append(exp)
            
            # Return all old experiences first, then current experiences
            return all_old_experiences + current_experiences
        
        # Collect all experiences from oldest to newest
        all_experiences = collect_all_experiences(experience)
        
        # Track loaded instructions to avoid duplicates
        loaded_instructions = set()
        new_experience = []
        
        # Format experiences in chronological order (oldest first)
        for msg in all_experiences:
            instruction = msg.metadata["instruction"]
            
            # Skip if this instruction has already been processed
            if instruction in loaded_instructions:
                continue
                
            # Add to tracking set
            loaded_instructions.add(instruction)
            
            # Format the experience
            new_experience.append(Message(content=instruction, role="user"))
            new_experience.append(Message(content=msg.content, role="assistant"))
            new_experience.append(Message(content=msg.metadata["output"], role="assistant"))
            
        return new_experience
    
    async def _think(self):
        pass
    
    async def _write_and_exec_code(self, max_retry: int = 3):
        return await super()._write_and_exec_code(max_retry)
    
    def load_model_usage(self, experiences: list[Message]) -> str:
        model_types = [
            "token-classification", "text-classification", "zero-shot-classification",
            "translation", "summarization", "question-answering", "text-generation",
            "sentence-similarity", "tabular-classification", "tabular-regression",
            "object-detection", "image-classification", "image-to-text",
            "automatic-speech-recognition", "audio-classification", "video-classification"
        ]
        exp_str = str(experiences)
        model_types_presense = [
          model_type for model_type in model_types if model_type in exp_str
        ]
        
        path = "prism/ext/mas/templates/model.json"
        with open(path, "r", encoding="utf-8") as file:
            model_data = json.load(file)
        model_description = ""
        for idx, (key, value) in enumerate(model_data.items()):
            if key in model_types_presense:
                model_description += f"{idx}. {key}. How to use:  with interface {value['interface']} and output {value['output']}.\n"
        return model_description
    
    
    async def run(self, experience: list[Message] = []) -> str:
        todo = self.ac.todo
        if not todo:
            raise Exception("No todo found!")
        logger.info(f"ready to {todo.name}")
        counter = 0
        max_retry = 3
        sample_data = self.data.iloc[0].to_dict()
        del sample_data["input_id"]
        del sample_data["label"]
        formatted_experience = self.format_experience(experience)
        for msg in formatted_experience:
            if msg.role == "user":
                print("User:", msg.content)
            elif msg.role == "assistant":
                print("Assistant:", msg.content)
            print("-" * 80)

        while counter < max_retry:
            counter += 1
            code = await todo.run(
                user_requirement=self.problem,
                memory=formatted_experience,
                prompt_template_manager=self.prompt_template_manager,
                working_memory=self.working_memory.get(),
                sample_data=describe_dict(sample_data),
                model_description=self.load_model_usage(experience)
            )
            self.working_memory.add(Message(content=code, role="assistant"))
            try:
                compile(code, "<generated_workflow>", "exec")
            except SyntaxError as se:
                err = (
                    f"[COMPILE ERROR] SyntaxError: {se.msg} "
                    f"(line {se.lineno}, col {se.offset})\n"
                    f"--> {se.text}"
                )
                self.working_memory.add(Message(content=err, role="user"))
                logger.error(err)
                continue 
            
            mod_name = "generated_workflow"
            spec = importlib.util.spec_from_loader(mod_name, loader=None)
            if spec is None:
                raise Exception("Failed to create module spec")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            try:
                exec(code, mod.__dict__)
                Workflow = getattr(mod, "Workflow")
                wf = Workflow()
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                err = f"[GENERATE ERROR] {error_type}: {error_msg}\n"
                err += "Exception while generating Workflow, You have to generate solution:\n"
                err += f"Full traceback:\n{_traceback_str(e)}"
                self.working_memory.add(Message(content=err, role="user"))
                logger.error(err)
                continue
            try:
                pred = await wf(sample_data)
                print("Prediction:", pred)
            except Exception as e:
                # Extract more specific error information
                error_type = type(e).__name__
                error_msg = str(e)
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                
                # Find the line number and code that caused the error
                error_line = "Unknown"
                error_code = "Unknown"
                for line in tb_lines:
                    if "line" in line and "in __call__" in line:
                        # Extract line number from traceback
                        import re
                        line_match = re.search(r'line (\d+)', line)
                        if line_match:
                            error_line = line_match.group(1)
                
                # Try to get the actual code line that caused the error
                try:
                    code_lines = code.split('\n')
                    if error_line != "Unknown" and int(error_line) <= len(code_lines):
                        error_code = code_lines[int(error_line) - 1].strip()
                except:
                    pass
                
                err = f"[RUN ERROR] {error_type}: {error_msg}\n"
                err += f"Error occurred at line {error_line}: {error_code}\n"
                err += f"Full traceback:\n{_traceback_str(e)}"
                
                self.working_memory.add(Message(content=err, role="user"))
                logger.error(err)
                continue
            break
        return code
    

if __name__ == "__main__":
    synthesizer = Synthesizer()
    print(synthesizer.load_model_usage(experiences=[Message(content="image-classification")]))