import os
import json
import glob
import datetime
from typing import List, Union
from bs4 import BeautifulSoup
import pandas as pd

from utils.common import read_json_file, write_json_file
from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.llm import LLM

class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.prompt_template_manager = PromptTemplateManager(template_dirs=[os.path.join(root_path, "prompts")])
        self.llm = LLM()
        
    def load_model_type_description(self, directory: str, models: Union[List[str], str]) -> str:
        file_path = os.path.join(directory, "model.json")
        if models == "<all>":
            models = list(read_json_file(os.path.join(directory, "model.json"), encoding="utf-8").keys())
        model_type_description = ""
        for id, model_name in enumerate(models):
            model_type_description += self._load_model_type_description(id + 1, model_name, file_path)
            model_type_description += "\n"
        return model_type_description
    
    def _load_model_type_description(self, id: int, model_name: str, file_path: str) -> str:
        model_data = read_json_file(file_path, encoding="utf-8")
        matched_data = model_data[model_name]
        desc = matched_data["description"]
        guidance = matched_data["guidance"]
        return f"""{id}. {model_name}
    - Description: {desc}. 
    - Usecase Guidance: {guidance}"""
    
    def load_agent_description(self, directory: str, agents: Union[List[str], str]) -> str:
        
        file_path = os.path.join(self.root_path, "agent.json")
        if agents == "<all>":
            agents = list(read_json_file(os.path.join(directory, "agent.json"), encoding="utf-8").keys())
        agent_description = ""
        for id, agent_name in enumerate(agents):
            agent_description += self._load_agent_description(id + 1, agent_name, file_path)
            agent_description += "\n"
        return agent_description
    
    def _load_agent_description(self, id: int, agent_name: str, file_path: str) -> str:
        agent_data = read_json_file(file_path, encoding="utf-8")
        matched_data = agent_data[agent_name]
        desc = matched_data["description"]
        guidance = matched_data["guidance"]
        return f"{id}. {agent_name}: {desc}. Guidance: {guidance}"
        
    def get_results_file_path(self, graph_path: str, reset: bool = False) -> str:
        path = os.path.join(graph_path, "results.json")
        if reset:
            if os.path.exists(path):
                os.remove(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_json_file(path, [], encoding="utf-8")
        else:
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                write_json_file(path, [], encoding="utf-8")
        return path
    
    def get_results(self, path: str) -> list:
        result_file = os.path.join(path, "results.json")
        data = read_json_file(result_file, encoding="utf-8")
        return data
    
    def create_result_data(self, round: int, score: float, cost: float,  generation_time: float, code: str) -> dict:
        now = datetime.datetime.now()
        return {
            "round": round,
            "score": score,
            "time": now,
            "cost": cost,
            "generation_time": generation_time,
            "code": code
        }
        
    def save_results(self, json_file_path: str, data: list):
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)
        
    def parse_xml(self, text: str, args: list[str], json_args: list[str] = []) -> dict:
        soup = BeautifulSoup(text, "html.parser")
        res = {}
        for arg in args:
            field = soup.find(arg)
            if not field:
                raise Exception(f"Field {arg} not found in the xml")
            res[arg] = field.get_text(strip=True)
        for json_arg in json_args:
            res[json_arg] = json.loads(res[json_arg])
            
        return res
        
    async def select_best_round(self, path: str, rounds: int, user_requirement: str) -> int:
        scores = self._load_scores(path)
        scores = {r: s for r, s in scores.items() if r <= rounds}
        if not scores:
            raise ValueError(f"No round <= {rounds} in {path}")
        best_score = max(scores.values())
        
        best_rounds = [r for r, s in scores.items() if s == best_score]
        if len(best_rounds) == 1:
            print("Best round is: ", best_rounds[0])
            return best_rounds[0]
        else:
            all_experiences = self._load_experiences(path)
            best_experiences = [r for r in all_experiences if r["round"] in best_rounds]
            solution_desc = self._format_solutions(best_experiences)
            sc_ensemble_input = self.prompt_template_manager.render(name="sc_ensemble", user_requirement=user_requirement, solutions=solution_desc)
            sc_ensemble_msg = self.llm.format_msg(sc_ensemble_input)
            sc_ensemble_rsp = await self.llm.aask(sc_ensemble_msg)
            sc_ensemble_rsp = self.parse_xml(sc_ensemble_rsp, ["thought", "solution_letter"])
            solution_letter = int(sc_ensemble_rsp["solution_letter"])
            return best_experiences[solution_letter - 1]["round"]
    
    def _format_solutions(self, results: list[dict]) -> str:
        desc = ""
        for id, result in enumerate(results):
            desc += f"""<Solution_{id + 1}>
            <code>{result["generated_code"]}</code>
            <logs_when_running_code>{result["logs"]}</logs_when_running_code>
</Solution_{id + 1}>"""
        return desc
    
    def _load_scores(self, path: str) -> dict:
        result_file = os.path.join(path, "results.json")
        data = read_json_file(result_file, encoding="utf-8")
        df = pd.DataFrame(data)
        scores_per_round = df.groupby("round")["score"].mean().to_dict()
        return scores_per_round
    
    def _load_experiences(self, path: str) -> list[dict]:
        pattern = os.path.join(path, "round_*", "experience.json")
        experience = []
        pattern = os.path.join(path, "round_*", "experience.json")
        for file in sorted(glob.glob(pattern)):
            try:
                data = read_json_file(file, encoding="utf-8")
                if isinstance(data, list):
                    experience.extend(data)
                else:
                    experience.append(data)
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
        return experience