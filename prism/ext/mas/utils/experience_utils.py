import os

from utils.common import read_json_file, write_json_file

class ExperienceUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        
    def load_experience(self, directory: str, round_number: int) -> dict:
        file_path = os.path.join(directory, f"round_{round_number}", "experience.json")
        
        experience = read_json_file(file_path, encoding="utf-8")
        return experience
    
    def load_all_experiences(self, directory: str, rounds: int) -> list[dict]:
        experiences = []
        for round_number in range(1, rounds + 1):
            experience = self.load_experience(directory, round_number)
            experiences.append(experience)
        return experiences
    
    def format_experience(self, experiences: list[dict]) -> str:
        formatted_experience = ""
        for experience in experiences:
            formatted_experience += f"""<round_{experience['round']}>
                <score>{experience['score']}</score>
                <sketch>{experience['sketch']}</sketch>
                <review>{experience["review"]}</review>
            </round_{experience['round']}>
            """
        return formatted_experience
    
    def update_experience(self, directory: str, round_number: int, experience: dict):
        write_json_file(os.path.join(directory, f"round_{round_number}", "experience.json"), experience, encoding="utf-8", indent=4)
    
    def create_experience_data(self, avg_score: float, sketch: str, workflow: str, logs: list[str], round: int, review: str, cost: float, generation_time: float, generated_code: str):
        return {
            "score": avg_score,
            "sketch": sketch,
            "workflow": workflow,
            "generated_code": generated_code,
            "logs": logs,
            "round": round,
            "review": review,
            "cost": cost,
            "generation_time": generation_time
        }