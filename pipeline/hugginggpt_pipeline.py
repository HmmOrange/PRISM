import os
import re
import json
import copy
import traceback
import yaml
import requests
from typing import Tuple
from queue import Queue
from bs4 import BeautifulSoup

from scripts.model_inference import model_inference

from templates.pipeline_template import Pipeline
from configs.models_config import ModelsConfig
from provider.llm_provider_registry import create_llm_instance
from utils.logs import logger
from utils.common import image_to_bytes
from utils.token_counter import TOKEN_MAX, ENCODINGS
from utils.constants import (
    COMPLETION_API_KEY, COMPLETION_BASE_URL,
    COMPLETION_MODEL_NAME, LOCAL_INFERENCE_ENDPOINT_URL,
    HUGGINGFACE_HEADERS
)
from utils.token_counter import TOKEN_COSTS


CONFIG_PATH = "configs/config.hugginggpt.yaml"
MODEL_PATH = "data/huggingface_models.jsonl"
API_ENDPOINT = COMPLETION_BASE_URL + "/chat/completions"
PROXY = None

HUGGINGGPT_CACHE_PATH = "data/hugginggpt_cache.json"

TASK_NAME_EXCLUDE_CACHE = ["dog_breed_or_cat_expression", "first_location_or_financial_status"]

config = yaml.load(open(CONFIG_PATH, "r"), Loader=yaml.FullLoader)

parse_task_prompt = config["prompt"]["parse_task"]
choose_model_prompt = config["prompt"]["choose_model"]
response_results_prompt = config["prompt"]["response_results"]

parse_task_tprompt = config["tprompt"]["parse_task"]
choose_model_tprompt = config["tprompt"]["choose_model"]
response_results_tprompt = config["tprompt"]["response_results"]

parse_task_demos_or_presteps = open(config["demos_or_presteps"]["parse_task"], "r").read()
choose_model_demos_or_presteps = open(config["demos_or_presteps"]["choose_model"], "r").read()
response_results_demos_or_presteps = open(config["demos_or_presteps"]["response_results"], "r").read()

MODELS = [json.loads(line) for line in open("data/huggingface_models.jsonl", "r", encoding="utf-8").readlines()]
MODELS_MAP = {}
for model in MODELS:
    tag = model["pipeline_tag"]
    if tag not in MODELS_MAP:
        MODELS_MAP[tag] = []
    MODELS_MAP[tag].append(model)
    
try:
    r = requests.get(LOCAL_INFERENCE_ENDPOINT_URL + "/running")
    if r.status_code != 200:
        raise ValueError("local inference endpoint is not running")
    else:
        logger.info("local inference endpoint is running")
except:
    raise ValueError("local inference endpoint is not running")

def get_choose_model_cache(task_name: str, tag: str):
    if not os.path.exists(HUGGINGGPT_CACHE_PATH):
        with open(HUGGINGGPT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(HUGGINGGPT_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.loads(f.read())
    task_cache = cache.get(task_name, None)
    if not task_cache:
        return None
    return task_cache.get(tag, None)

def save_choose_model_cache(task_name: str, tag: str, choose_model_str: str):
    if not os.path.exists(HUGGINGGPT_CACHE_PATH):
        with open(HUGGINGGPT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(HUGGINGGPT_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.loads(f.read())
    if task_name not in cache:
        cache[task_name] = {}
    cache[task_name][tag] = choose_model_str
    with open(HUGGINGGPT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def count_tokens(model_name: str, text: str) -> int:
    return len(ENCODINGS[model_name].encode(text))

def get_max_context_length(model_name: str) -> int:
    return TOKEN_MAX[model_name]

def get_token_ids_for_task_parsing(model_name):
    text = '''{"task": "token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification", "args", "text", "path", "dep", "id", "<GENERATED>-"}'''
    res = ENCODINGS[model_name].encode(text)
    res = list(set(res))
    return res

def get_token_ids_for_choose_model(model_name):
    text = '''{"id": "reason"}'''
    res = ENCODINGS[model_name].encode(text)
    res = list(set(res))
    return res

def parse_xml(text: str, args: list[str], json_args: list[str] = []) -> dict:
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

task_parsing_highlight_ids = get_token_ids_for_task_parsing(COMPLETION_MODEL_NAME)
choose_model_highlight_ids = get_token_ids_for_choose_model(COMPLETION_MODEL_NAME)

class CostManager:
    total_cost = 0

    def update_total_cost(self, cost: float):
        self.total_cost += cost

    def get_total_cost(self):
        return self.total_cost
    
    def reset_total_cost(self):
        self.total_cost = 0
        
cost_manager = CostManager()

def send_request(data: dict):
    api_key = data.pop("api_key")
    api_endpoint = data.pop("api_endpoint")
    HEADER = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(api_endpoint, json=data, headers=HEADER, proxies=PROXY)
    res_json = response.json()
    if "error" in res_json:
        return res_json
    logger.debug(response.text.strip())
    
    usage = res_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    cost = (prompt_tokens * TOKEN_COSTS[COMPLETION_MODEL_NAME]["prompt"] + completion_tokens * TOKEN_COSTS[COMPLETION_MODEL_NAME]["completion"]) / 1000
    cost_manager.update_total_cost(cost)
    logger.info(f"Total prompt tokens: {prompt_tokens} | Total completion tokens: {completion_tokens} |  Running cost: ${cost:.3f} | Total running cost: ${cost_manager.get_total_cost():.3f}")
    
    return res_json["choices"][0]["message"]["content"].strip()
    
def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'").replace('\n', ""))
    return text

def find_json(s):
    s = s.replace("\'", "\"")
    start = s.find("{")
    end = s.rfind("}")
    res = s[start:end+1]
    res = res.replace("\n", "")
    return res

def field_extract(s, field):
    try:
        field_rep = re.compile(f'{field}.*?:.*?"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    except:
        field_rep = re.compile(f'{field}:\ *"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    return extracted

def get_id_reason(choose_str):
    reason = field_extract(choose_str, "reason")
    id = field_extract(choose_str, "id")
    choose = {"id": id, "reason": reason}
    return id.strip(), reason.strip(), choose

def chitchat(messages):
    data = {
        "model": COMPLETION_MODEL_NAME,
        "messages": messages,
        "api_endpoint": API_ENDPOINT
    }
    return send_request(data)

def parse_task(context: list, input: str, api_key: str, api_endpoint: str, data: dict):
    demos_or_presteps = parse_task_demos_or_presteps
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": parse_task_tprompt})
    
    start = 0
    while start <= len(context):
        history = context[start:]
        prompt = replace_slot(parse_task_prompt, {
            "input": input,
            "context": history,
            "data": data
        })
        messages.append({"role": "user", "content": prompt})
        history_text = "<im_end>\nuser<im_start>".join([m["content"] for m in messages])
        num = count_tokens(COMPLETION_MODEL_NAME, history_text)
        if get_max_context_length(COMPLETION_MODEL_NAME) - num > 800:
            break
        messages.pop()
        start += 2
    logger.debug(messages)
    
    data = {
        "model": COMPLETION_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "logit_bias": {item: config["logit_bias"]["parse_task"] for item in task_parsing_highlight_ids},
        "api_endpoint": api_endpoint,
        "api_key": api_key
    }
    return send_request(data)
    
def choose_model(input: str, task: dict, metas: list, api_key: str, api_endpoint: str):
    prompt = replace_slot(choose_model_prompt, {
        "input": input,
        "task": task,
        "metas": metas,
    })
    demos_or_presteps = replace_slot(choose_model_demos_or_presteps, {
        "input": input,
        "task": task,
        "metas": metas
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": choose_model_tprompt})
    messages.append({"role": "user", "content": prompt})

    logger.debug(messages)
    data = {
        "model": COMPLETION_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "logit_bias": {item: config["logit_bias"]["choose_model"] for item in choose_model_highlight_ids},
        "api_endpoint": api_endpoint,
        "api_key": api_key
    }
    return send_request(data)
    
    
def get_model_status(model_id: str, url: str, headers,  queue = None):
    endpoint_type = "huggingface" if "huggingface" in url else "local"
    if "huggingface" in url:
        r = requests.get(url, headers=headers, proxies=PROXY)
    else:
        r = requests.get(url)
    if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
        if queue:
            queue.put((model_id, True, endpoint_type))
        return True
    else:
        if queue:
            queue.put((model_id, False, None))
        return False

def get_available_models(candidates: list, topk=5):
    all_available_models = {"local": [], "huggingface": []}
    threads = []
    result_queue = Queue()
    
    for candidate in candidates:
        all_available_models["local"].append(candidate["id"])
    
    return all_available_models        

def collect_result(command, choose, inference_result):
    result = {"task": command}
    result["inference result"] = inference_result
    result["choose model result"] = choose
    logger.debug(f"inference result: {inference_result}")
    return result

def run_task(input: str, command: dict, results: dict, api_key: str, api_endpoint: str, task_name: str):
    id = command["id"]
    args = command["args"]
    task = command["task"]
    deps = command["dep"]
    if deps[0] != -1:
        dep_tasks = [results[dep] for dep in deps]
    else:
        dep_tasks = []
        
    logger.info(f"Run task: {id} - {task}")
    logger.info("Deps: " + json.dumps(dep_tasks))
    
    
    if deps[0] != -1:
        if "image" in args and "<GENERATED>-" in args["image"]:
            resource_id = int(args["image"].split("-")[1])
            if "generated image" in results[resource_id]["inference result"]:
                args["image"] = results[resource_id]["inference result"]["generated image"]
        if "audio" in args and "<GENERATED>-" in args["audio"]:
            resource_id = int(args["audio"].split("-")[1])
            if "generated audio" in results[resource_id]["inference result"]:
                args["audio"] = results[resource_id]["inference result"]["generated audio"]
        if "text" in args and "<GENERATED>-" in args["text"]:
            resource_id = int(args["text"].split("-")[1])
            if "generated text" in results[resource_id]["inference result"]:
                args["text"] = results[resource_id]["inference result"]["generated text"]
    
    text = image = audio = None
    
    for dep_task in dep_tasks:
        if "generated text" in dep_task["inference result"]:
            text = dep_task["inference result"]["generated text"]
            logger.info("Detect the generated text from the dependency task (from results):" + text)
        elif "text" in dep_task["task"]["args"]:
            text = dep_task["task"]["args"]["text"]
            logger.info("Detect the text of dependency task (from args): " + text)
        if "generated image" in dep_task["inference result"]:
            image = dep_task["inference result"]["generated image"]
            logger.info("Detect the generated image from the dependency task (from results): " + image)
        elif "image" in dep_task["task"]["args"]:
            image = dep_task["task"]["args"]["image"]
            logger.info("Detect the image of dependency task (from args): " + image)
        if "generated audio" in dep_task["inference result"]:
            audio = dep_task["inference result"]["generated audio"]
            logger.info("Detect the generated audio from the dependency task (from results): " + audio)
        elif "audio" in dep_task["task"]["args"]:
            audio = dep_task["task"]["args"]["audio"]
            logger.info("Detect the audio of dependency task (from args): " + audio)
        
    if "image" in args and "<GENERATED>" in args["image"]:
        if image:
            args["image"] = image
    if "audio" in args and "<GENERATED>" in args["audio"]:
        if audio:
            args["audio"] = audio
    if "text" in args and "<GENERATED>" in args["text"]:
        if text:
            args["text"] = text
        
    command["args"] = args
    
    logger.info(f"args: {args}")
    
    if task not in MODELS_MAP:
        logger.warning(f"no available models on {task} task.")
        return False
    
    candidates = MODELS_MAP[task]
    all_available_models = get_available_models(candidates, config["num_candidate_models"])
    all_available_model_ids = all_available_models["local"] + all_available_models["huggingface"]
    logger.info(f"all available model ids: {all_available_model_ids}")
    
    if len(all_available_model_ids) == 0:
        logger.warning(f"no available models on {command['task']}")
        inference_result = {"error": f"no available models on {command['task']} task."}
        results[id] = collect_result(command, "", inference_result)
        return False
    
    if len(all_available_model_ids) == 1:
        best_model_id = all_available_model_ids[0]
        hosted_on = "local" if best_model_id in all_available_models["local"] else "huggingface"
        reason = "Only one model available."
        choose = {"id": best_model_id, "reason": reason}
        logger.info(f"chosen model: {choose}")
    else:
        cand_models_info = [
            {
                "id": model["id"],
                "inference endpoint": all_available_models.get(
                    "local" if model["id"] in all_available_models["local"] else "huggingface"
                ),
                "likes": model.get("likes"),
                "description": model.get("description", "")[:config["max_description_length"]],
                # "language": model.get("meta").get("language") if model.get("meta") else None,
                "tags": model.get("meta").get("tags") if model.get("meta") else None,
            }
            for model in candidates
            if model["id"] in all_available_model_ids
        ]
        cache_choose_str = get_choose_model_cache(task_name, task)
        if False:
            choose_str = cache_choose_str
        else:
            choose_str = choose_model(input, command, cand_models_info, api_key, api_endpoint)
            # save_choose_model_cache(task_name, task, choose_str)
        
        logger.info(f"chosen model: {choose_str}")
        try:
            choose = json.loads(choose_str)
            reason = choose["reason"]
            best_model_id = choose["id"]
            hosted_on = "local" if best_model_id in all_available_models["local"] else "huggingface"
        except:
            logger.warning(f"the response [ {choose_str} ] is not a valid JSON, try to find the model id and reason in the response.")
            choose_str = find_json(choose_str)
            best_model_id, reason, choose  = get_id_reason(choose_str)
            hosted_on = "local" if best_model_id in all_available_models["local"] else "huggingface"
    try:        
        inference_result = model_inference(best_model_id, args, hosted_on, command['task'], return_resource=True)
    except:
        inference_result = {"error": f"Inference error: {traceback.format_exc()}"}
    if "error" in inference_result:
        results[id] = collect_result(command, choose, inference_result)
        return False
      
    results[id] = collect_result(command, choose, inference_result)
    
    return True
            
def fix_dep(tasks):
    for task in tasks:
        args = task["args"]
        task["dep"] = []
        for k, v in args.items():
            if isinstance(v, str) and "<GENERATED>" in v:
                dep_task_id = int(v.split("-")[1])
                if dep_task_id not in task["dep"]:
                    task["dep"].append(dep_task_id)
        if len(task["dep"]) == 0:
            task["dep"] = [-1]
    return tasks

def unfold(tasks):
    flag_unfold_task = False
    try:
        for task in tasks:
            for key, value in task["args"].items():
                if isinstance(value, str) and "<GENERATED>" in value:
                    generated_items = value.split(",")
                    if len(generated_items) > 1:
                        flag_unfold_task = True
                        for item in generated_items:
                            new_task = copy.deepcopy(task)
                            dep_task_id = int(item.split("-")[1])
                            new_task["dep"] = [dep_task_id]
                            new_task["args"][key] = item
                            tasks.append(new_task)
                        tasks.remove(task)
    except Exception as e:
        print(e)
        traceback.print_exc()
        logger.debug("unfold task failed.")

    if flag_unfold_task:
        logger.debug(f"unfold tasks: {tasks}")
        
    return tasks

def response_results(input, results, api_key, api_endpoint: str):
    results = [v for k,v  in sorted(results.items(), key=lambda item: item[0])]
    prompt = replace_slot(response_results_prompt, {
        "input": input
    })
    demos_or_presteps = replace_slot(response_results_demos_or_presteps, {
        "input": input,
        "processes": results
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": response_results_tprompt})
    messages.append({"role": "user", "content": prompt})
    logger.debug(messages)
    data = {
        "model": COMPLETION_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "api_key": api_key,
        "api_endpoint": api_endpoint
    }
    xml_response = send_request(data)
    try:
        response = parse_xml(xml_response, ["prediction"])["prediction"]
    except:
        response = xml_response
        
    try:
        response = json.loads(response)
    except:
        pass
    try:
        return response.strip()
    except:
        return response

class HuggingGPTPipeline(Pipeline):    
        
    async def __call__(self, task_description: str, data: dict) -> Tuple[str, float]:
        task_name = data["task_name"]
        with open("templates/hugginggpt_solution.py", "r", encoding="utf-8") as f:
            solution = f.read()
        solution = solution.replace("<<<TASK_DESCRIPTION>>>", task_description)
        solution = solution.replace("<<<TASK_NAME>>>", task_name)
        
        return solution, 0
    
    
import threading
import time

if __name__ == '__main__':
#     user_requirement = """Given a single district record with demographic, economic, and geographic features, predict the median house value for that district.

# The record includes:

# - MedInc: Median income in the district (float).
# - HouseAge: Median house age in the district (float).
# - AveRooms: Average number of rooms per household (float).
# - AveBedrms: Average number of bedrooms per household (float).
# - Population: Total population in the district (float).
# - AveOccup: Average household occupancy (float).
# - Latitude: Geographic latitude (float).
# - Longitude: Geographic longitude (float).

# Output must be a single regression prediction: Estimated median house value (float)."""
#     data = {
#         "text_data": {
#             "MedInc": 3.5104,
#             "HouseAge": 29.0,
#             "AveRooms": 26.465968586387433,
#             "AveBedrms": 5.424083769633508,
#             "Population": 410.0,
#             "AveOccup": 2.1465968586387434,
#             "Latitude": 34.3,
#             "Longitude": -116.99
#         }
#     }
    user_requirement = """Given a low-resolution image (32x32), classify it into one of the 100 categories from the CIFAR-100 taxonomy.
"""
    data = {
        "image_paths": ["tasks/node-level/100_classification/validation/inputs/1/image.png"]
    }
    task_name = "100_classification"
    messages = [{"role": "user", "content": user_requirement}]
    context = messages[:-1]
    input = messages[-1]["content"]
    
    task_str = parse_task(context, input, COMPLETION_API_KEY, API_ENDPOINT, data)
    task_str = task_str.strip()
    logger.info(task_str)
    
    try:
        tasks = json.loads(task_str)
    except Exception as e:
        raise e
    
    tasks = unfold(tasks)
    tasks = fix_dep(tasks)
    logger.debug(tasks)
    
    results = {}
    threads = []
    tasks = tasks[:]
    d = dict()
    retry = 0
    while True:
        num_thread = len(threads)
        for task in tasks:
            for dep_id in task["dep"]:
                if dep_id >= task["id"]:
                    task["dep"] = [-1]
                    break
            dep = task["dep"]
            if dep[0] == -1 or len(list(set(dep).intersection(d.keys()))) == len(dep):
                tasks.remove(task)
                thread = threading.Thread(target=run_task, args=(input, task, d, COMPLETION_API_KEY, API_ENDPOINT, task_name))
                thread.start()
                threads.append(thread)
        if num_thread == len(threads):
            time.sleep(0.5)
            retry += 1
        if retry > 160:
            logger.debug("User has waited too long, Loop break.")
            break
        if len(tasks) == 0:
            break
        
    for thread in threads:
        thread.join()    
    
    results = d.copy()

    logger.info(results)
    
    response = response_results(input, results, COMPLETION_API_KEY, API_ENDPOINT)
    
    logger.info(f"response: {response}")