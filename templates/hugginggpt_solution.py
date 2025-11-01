import json
import threading
import time


from utils.logs import logger
from utils.constants import (
    COMPLETION_API_KEY
)
from pipeline.hugginggpt_pipeline import (
    parse_task, API_ENDPOINT, unfold,
    fix_dep, run_task, response_results, cost_manager
)

class Workflow:
    def __init__(self):
        self.task_description = '''<<<TASK_DESCRIPTION>>>'''
        self.task_name = '''<<<TASK_NAME>>>'''
        cost_manager.reset_total_cost()
        
    def save_total_cost(self):
        try:
            with open("data/hugginggpt_cost.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}
        total_cost = cost_manager.get_total_cost()
        data[self.task_name] = total_cost
        with open("data/hugginggpt_cost.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
        
    async def __call__(self, data: dict):
        messages = [{"role": "user", "content": self.task_description}]
        context = messages[:-1]
        input = messages[-1]["content"]
        
        task_str = parse_task(context, input, COMPLETION_API_KEY, API_ENDPOINT, data)
        if "error" in task_str:
            self.save_total_cost()
            return ""
 
        task_str = task_str.strip()
        logger.info(task_str)
        
        try:
            tasks = json.loads(task_str)
        except Exception as e:
            logger.error(e)
            self.save_total_cost()
            return ""

        if task_str == "[]":
            self.save_total_cost()
            return ""
        
        tasks = unfold(tasks)
        tasks = fix_dep(tasks)
        logger.debug(tasks)
        
        results = {}
        threads = []
        tasks = tasks[:]
        try:
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
                        thread = threading.Thread(target=run_task, args=(input, task, d, COMPLETION_API_KEY,  API_ENDPOINT, self.task_name))
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
        except:
            response = ""
        logger.info(f"response: {response}")
        
        self.save_total_cost()
        return response