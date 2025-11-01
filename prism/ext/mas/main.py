import json

from prism.llm import LLM

def load_model_usage():
    model_description = ""
    with open("prism/ext/mas/templates/model.json", "r", encoding="utf-8") as file:
        model_data = json.load(file)
    for id, (model_name, model) in enumerate(model_data.items()):
        model_description += f"{id}. {model_name}: {model['description']}. Guidance: {model['guidance']}\n"
    return model_description

def load_agent_usage():
    agent_description = ""
    with open("prism/ext/mas/templates/agent.json", "r", encoding="utf-8") as file:
        agent_data = json.load(file)
        
def search():
    pass

if __name__ == '__main__':
    pass