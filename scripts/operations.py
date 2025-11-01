import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import pandas as pd

from utils.logs import logger
from scripts.calculator import Calculator, MetricType
from scripts.data import (
    load_data, load_metadata, load_task_description, 
    load_label, load_prediction, load_task_dirs
)
from scripts.utils import get_save_task_dir, save_csv
from scripts.workflows import load_pipeline, load_workflow, write_workflow



async def generate_workflow_for_task(task_dir, save_dir, pipeline_path, rounds, p_name):
    """Generate workflow for a single task"""
    total_generation_time = 0
    total_generation_cost = 0
    task_name = Path(task_dir).name
    results = []
    
    for round in tqdm(range(rounds), leave=False):
        while task_dir.endswith("/"):
            task_dir = task_dir[:-1]
        tqdm.write(f"[{p_name}] Generate workflow: {task_name} for round: {round}")
        validation_data = load_data(os.path.join(task_dir, "validation"))
        label = load_label((os.path.join(task_dir, "validation")))
        metadata = load_metadata(task_dir)
        level = task_dir.split("/")[-2]

        task_description = load_task_description(task_dir)
        data = {
            "validation_data": pd.DataFrame(validation_data),
            "label": label,
            "task_name": task_name,
            "metric": metadata["metric"],
            "level": level
        }
        pipeline = load_pipeline(pipeline_path)
        generation_time_start = time.time()
        try:
            workflow, cost = await pipeline(task_description, data)
            if cost is None or not isinstance(cost, (int, float)):
                cost = 0.0
            else:
                cost = float(cost)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error generating workflow: {e}")
            workflow = ""
            cost = 0.0
        generation_time_end = time.time()
        generation_time = generation_time_end - generation_time_start
        total_generation_time += generation_time
        total_generation_cost += cost
        tqdm.write(f"[{p_name}] Generation time: {generation_time} seconds, Cost: {cost}")
        round_num = round if rounds > 1 else None
        save_task_dir = get_save_task_dir(task_dir, save_dir)
        write_workflow(save_task_dir, workflow, round_num=round_num)
        results.append({
            "id": f"{task_name}_round_{round_num}" if round_num is not None else task_name,
            "time": generation_time,
            "cost": cost
        })
    save_csv(save_task_dir, "generation_cost.csv", pd.DataFrame(results))
    # Calculate averages and update scores.csv
    avg_generation_time = total_generation_time / rounds
    avg_generation_cost = total_generation_cost / rounds

    tqdm.write(f"[{p_name}] Average generation time: {avg_generation_time} seconds, Average cost: {avg_generation_cost}")


async def run_workflow_for_task(task_dir, save_dir, rounds, p_name):
    """Run workflow for a single task"""
    inputs = load_data(os.path.join(task_dir, "test"))
    task_name = Path(task_dir).name
    total_execution_times = []
    task_description = load_task_description(task_dir)
    for round in tqdm(range(rounds), leave=False):
        tqdm.write(f"[{p_name}] Run workflow: {task_name} for round: {round}")
        round_num = round if rounds > 1 else None
        predictions = []
        save_task_dir = get_save_task_dir(task_dir, save_dir)
        total_execution_time = 0
        try:
            workflow = load_workflow(save_task_dir, round_num=round_num)
            try:
                # Run workflow with first input for setup model server
                await workflow(inputs[0])
            except Exception as e:
                pass
            for input in tqdm(inputs, leave=False):
                input_id = input["input_id"]
                tqdm.write(f"[{p_name}] Run workflow: {task_name} for round: {round} for input: {input_id}")
                input_execution_time_start = time.time()
                try:
                    prediction = await workflow(input)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Error running workflow: {e}")
                    prediction = ""
                input_execution_time_end = time.time()
                input_execution_time = input_execution_time_end - input_execution_time_start
                tqdm.write(f"[{p_name}] Input execution time: {input_execution_time} seconds")
                total_execution_time += input_execution_time
                predictions.append({"id": input_id, "prediction": prediction, "execution_time": input_execution_time})
            
            total_execution_times.append(total_execution_time)
            tqdm.write(f"[{p_name}] Total execution time for round {round}: {total_execution_time} seconds")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error loading workflow: {e}")
            workflow = None
            for input in inputs:
                predictions.append({"id": input["input_id"], "prediction": "", "execution_time": 0})
            total_execution_times.append(0)
        
        # Save predictions with execution times
        if round_num is not None:
            filename = f"predictions_{round_num}.csv"
        else:
            filename = f"predictions.csv"
        save_csv(save_task_dir, filename, pd.DataFrame(predictions))



def calculate_scores_for_task(task_dir, save_dir, rounds, p_name):
    """Calculate scores for a single task"""
    tqdm.write(f"[{p_name}] Calculate score: {Path(task_dir).name}")
    metadata = load_metadata(task_dir)
    task_name = Path(task_dir).name
    metric = metadata["metric"]
    metric_type = MetricType(metric)
    test_dir = os.path.join(task_dir, "test")
    print(test_dir)
    label = load_label(test_dir)
    df_scores = []
    calculator = Calculator()
    
    for round in range(rounds):
        round_num = round if rounds > 1 else None
        save_task_dir = get_save_task_dir(task_dir, save_dir)
        prediction = load_prediction(save_task_dir, round_num)
        _, df_score = calculator(metric_type, prediction, label)
        df_scores.append(df_score)
        
        # Save individual round scores if multiple rounds
        if round_num is not None:
            filename = f"scores_{round_num}.csv"
        else:
            filename = f"scores.csv"
            
        save_csv(save_task_dir, filename, df_score)
    
    df_avg_score = df_scores[0].copy()
    for col in df_avg_score.columns:
        if col == "score" and pd.api.types.is_numeric_dtype(df_avg_score[col]):
            col_sum = df_avg_score[col].copy()
            for df in df_scores[1:]:
                if col in df.columns:
                    col_sum += df[col]
            df_avg_score[col] = col_sum / len(df_scores)
        if col == "execution_time" and pd.api.types.is_numeric_dtype(df_avg_score[col]):
            col_sum = df_avg_score[col].copy()
            for df in df_scores[1:]:
                if col in df.columns:
                    col_sum += df[col]
            df_avg_score[col] = col_sum / len(df_scores)
    if round_num:
        save_csv(save_task_dir, f"avg_scores.csv", df_avg_score)
    # Update average score in scores.csv
    save_task_dir = get_save_task_dir(task_dir, save_dir)
    avg_score = df_avg_score["score"].mean()
    avg_execution_time = df_avg_score["execution_time"].mean()
    generation_cost_df = pd.read_csv(os.path.join(save_task_dir, "generation_cost.csv"))
    generation_cost_df["cost"] = generation_cost_df["cost"].astype(float)
    generation_cost_df["time"] = generation_cost_df["time"].astype(float)
    avg_generation_cost = generation_cost_df["cost"].mean()
    avg_generation_time = generation_cost_df["time"].mean()
    return {
        "id": task_name,
        "score": avg_score,
        "execution_time": avg_execution_time,
        "generation_cost": avg_generation_cost,
        "generation_time": avg_generation_time
    }


def calculate_scores(paths, save_dir, rounds):
    """Calculate scores for given paths"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        final_score = []
        for task_dir in tqdm(task_dirs, leave=False):
            score_result = calculate_scores_for_task(task_dir, save_dir, rounds, Path(p).name)
            final_score.append(score_result)
        print("Average score: ", pd.DataFrame(final_score)["score"].mean())
        save_p = get_save_task_dir(p, save_dir)
        save_csv(save_p, "final_scores.csv", pd.DataFrame(final_score))


async def generate_and_run(paths, save_dir, pipeline_path, rounds, run_after=False, calculate_after=False):
    """Generate workflows and optionally run and calculate for each task sequentially"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        final_scores = []
        
        for task_dir in tqdm(task_dirs, leave=False):
            tqdm.write(f"[{Path(p).name}] Processing task: {Path(task_dir).name}")
            
            # Step 1: Generate workflow for this task
            await generate_workflow_for_task(task_dir, save_dir, pipeline_path, rounds, Path(p).name)
            
            # Step 2: Run workflow if requested
            if run_after:
                print(f"Running workflow for {Path(task_dir).name}...")
                await run_workflow_for_task(task_dir, save_dir, rounds, Path(p).name)
                
                # Step 3: Calculate scores if requested
                if calculate_after:
                    print(f"Calculating scores for {Path(task_dir).name}...")
                    score_result = calculate_scores_for_task(task_dir, save_dir, rounds, Path(p).name)
                    final_scores.append(score_result)
        
        # Save final scores if we calculated them
        if calculate_after and run_after and final_scores:
            print("Average score: ", pd.DataFrame(final_scores)["score"].mean())
            save_p = get_save_task_dir(p, save_dir)
            save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_scores))


async def run_and_calculate(paths, save_dir, rounds, calculate_after=False):
    """Run workflows and optionally calculate scores for each task sequentially"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        final_scores = []
        
        for task_dir in tqdm(task_dirs, leave=False):
            tqdm.write(f"[{Path(p).name}] Processing task: {Path(task_dir).name}")
            
            # Step 1: Run workflow for this task
            await run_workflow_for_task(task_dir, save_dir, rounds, Path(p).name)
            
            # Step 2: Calculate scores if requested
            if calculate_after:
                print(f"Calculating scores for {Path(task_dir).name}...")
                score_result = calculate_scores_for_task(task_dir, save_dir, rounds, Path(p).name)
                final_scores.append(score_result)
        
        # Save final scores if we calculated them
        if calculate_after and final_scores:
            print("Average score: ", pd.DataFrame(final_scores)["score"].mean())
            save_p = get_save_task_dir(p, save_dir)
            save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_scores)) 