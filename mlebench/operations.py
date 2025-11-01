import importlib
import importlib.util
import os
from pathlib import Path
from tqdm import tqdm
from typing import Union, Dict, Any, Callable, Tuple, List
import pandas as pd
import asyncio
import time

from utils.logs import logger
from mlebench.grade import grade_csv
from mlebench.registry import registry
from scripts.workflows import load_pipeline, write_workflow, load_workflow
from scripts.data import load_task_description, load_task_dirs
from scripts.utils import get_save_task_dir, save_csv


def load_competition_module(competition_dir: Union[str, Path]) -> Any:
    competition_dir = Path(competition_dir).resolve()
    module_path = competition_dir / "load_mapping_data.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Not found: {module_path}")

    spec = importlib.util.spec_from_file_location("load_mapping_data", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_competition_data(
    competition_dir: Union[str, Path],
) -> Tuple[List, List, pd.DataFrame, pd.DataFrame]:
    """
    Load train/test data by dynamically importing `<competition_dir>/load_mapping_data.py`.

    Returns:
        {"train": <any>, "test": <any>}
    """
    module = load_competition_module(competition_dir)
    # Expect the module to define these two functions
    if not hasattr(module, "load_train_data") or not hasattr(module, "load_test_data"):
        raise AttributeError(
            "Expected functions 'load_train_data' and 'load_test_data' in load_mapping_data.py"
        )

    # If your functions take no arguments, remove `competition_dir` below.
    train_data, train_label = module.load_train_data(competition_dir)
    test_data, test_label = module.load_test_data(competition_dir)

    return train_data, test_data, train_label, test_label

def load_competition_train_data(
    competition_dir: Union[str, Path],
):
    module = load_competition_module(competition_dir)
    if not hasattr(module, "load_train_data"):
        raise AttributeError("Expected function 'load_train_data' in load_mapping_data.py")
    return module.load_train_data(competition_dir)
    
def load_competition_test_data(
    competition_dir: Union[str, Path],
):
    module = load_competition_module(competition_dir)
    if not hasattr(module, "load_test_data"):
        raise AttributeError("Expected function 'load_test_data' in load_mapping_data.py")
    return module.load_test_data(competition_dir)

def load_competition_lite_test_data(
    competition_dir: Union[str, Path]
):
    module = load_competition_module(competition_dir)
    if not hasattr(module, "load_lite_test_data"):
        raise AttributeError("Expected function 'load_lite_test_data' in load_mapping_data.py")
    return module.load_lite_test_data(competition_dir)


async def generate_workflow_for_competition(
    competition_dir: str,
    save_dir: str,
    pipeline_path: str,
    lite: bool,
    rounds: int,
    competition_id: str,
):
    total_generation_time = 0
    total_generation_cost = 0
    results = []
    for round in tqdm(range(rounds), leave=False):
        tqdm.write(
            f"Competition [{competition_id}] Generate workflow: {Path(competition_dir).name} for round: {round}"
        )
        train_data, train_label = load_competition_train_data(
            competition_dir
        )
        data = {
            "validation_data": pd.DataFrame(train_data),
            "task_name": competition_id,
            "level": "mlebench",
        }
        task_description = load_task_description(competition_dir)
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
        tqdm.write(
            f"[{competition_id}] Generation time: {generation_time} seconds, Cost: {cost}"
        )
        round_num = round if rounds > 1 else None
        save_competition_dir = get_save_task_dir(competition_dir, save_dir, lite)
        write_workflow(save_competition_dir, workflow, round_num=round_num)
        results.append(
            {
                "id": (
                    f"{competition_id}_round_{round_num}"
                    if round_num is not None
                    else competition_id
                ),
                "time": generation_time,
                "cost": cost,
            }
        )
    save_csv(save_competition_dir, "generation_cost.csv", pd.DataFrame(results))
    avg_generation_time = total_generation_time / rounds
    avg_generation_cost = total_generation_cost / rounds

    tqdm.write(
        f"[{competition_id}] Average generation time: {avg_generation_time} seconds, Average cost: {avg_generation_cost}"
    )


async def process_single_input(
    semaphore: asyncio.Semaphore,
    workflow: Callable,
    input_data: Dict[str, Any],
    competition_id: str,
    competition_dir: str,
    round: int,
) -> Dict[str, Any]:
    """Xử lý một input data với semaphore để kiểm soát đồng thời"""
    async with semaphore:

        input_id = input_data["input_id"]
        fields = input_data.get("fields", {})
        predict_keys = input_data.get("predict_keys", [])
        tqdm.write(
            f"[{competition_id}] Run workflow: {Path(competition_dir).name} for round: {round} for input: {input_id}"
        )

        prediction_result = {}
        execution_time_start = time.time()
        try:
            prediction = await workflow(input_data)
            if not isinstance(prediction, list):
                prediction = [prediction]
        except Exception as e:
            # import traceback

            # traceback.print_exc()
            logger.error(f"Error running workflow: {e}")
            prediction = []
        execution_time_end = time.time()

        if len(prediction) != len(predict_keys):
            prediction = [""] * len(predict_keys)

        prediction_result = {k: v for k, v in zip(predict_keys, prediction)}

        execution_time = execution_time_end - execution_time_start
        tqdm.write(f"[{competition_id}] Execution time: {execution_time} seconds")
        return {**fields, **prediction_result, "execution_time": execution_time}


async def run_workflow_for_competition(
    competition_dir: str,
    save_dir: str,
    lite: bool,
    rounds: int,
    competition_id: str,
    max_concurrent: int,
):
    module = load_competition_module(competition_dir)
    if not hasattr(module, "mapping"):
        raise AttributeError("Expected functions 'mapping' in load_mapping_data.py")
    mapping = module.mapping
    for round in tqdm(range(rounds), leave=False):
        tqdm.write(
            f"Competition [{competition_id}] Run workflow: {Path(competition_dir).name} for round: {round}"
        )
        round_num = round if rounds > 1 else None
        if lite:
            test_data, test_label = load_competition_lite_test_data(
                competition_dir
            )
        else:
            test_data, test_label = load_competition_test_data(
                competition_dir
            )
        save_competition_dir = get_save_task_dir(competition_dir, save_dir, lite)
        workflow = load_workflow(save_competition_dir, round_num=round_num)
        # Create semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all test_data
        tasks = []

        try:  # run workflow with first input to setup model server
            await workflow(test_data[0])
        except Exception as e:
            pass

        for input_data in test_data:
            task = process_single_input(
                semaphore, workflow, input_data, competition_id, competition_dir, round
            )
            tasks.append(task)

        # Run all tasks concurrently with progress bar
        tqdm.write(
            f"[{competition_id}] Processing {len(tasks)} inputs with max {max_concurrent} concurrent tasks"
        )
        predictions = []

        # Use a different approach with progress tracking
        try:
            # Create progress bar
            pbar = tqdm(
                total=len(tasks), desc=f"[{competition_id}] Round {round}", leave=False
            )

            # Wrap each task with progress callback
            async def task_with_progress(task, index):
                try:
                    result = await task
                    pbar.update(1)
                    return result
                except Exception as e:
                    logger.error(f"Task {index} failed: {e}")
                    # Create empty prediction for failed task
                    input_data = test_data[index]
                    fields = input_data.get("fields", {})
                    predict_keys = input_data.get("predict_keys", [])
                    prediction_result = {k: "" for k in predict_keys}
                    pbar.update(1)
                    return {**fields, **prediction_result}

            # Create wrapped tasks
            wrapped_tasks = [
                task_with_progress(task, i) for i, task in enumerate(tasks)
            ]

            # Run all tasks and collect results
            results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
            pbar.close()

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                    # Create empty prediction for failed task
                    input_data = test_data[i]
                    fields = input_data.get("fields", {})
                    predict_keys = input_data.get("predict_keys", [])
                    prediction_result = {k: "" for k in predict_keys}
                    predictions.append({**fields, **prediction_result})
                else:
                    predictions.append(result)

        except Exception as e:
            logger.error(f"Error in concurrent processing: {e}")
            # Fallback to empty predictions
            predictions = []
            for input_data in test_data:
                fields = input_data.get("fields", {})
                predict_keys = input_data.get("predict_keys", [])
                prediction_result = {k: "" for k in predict_keys}
                predictions.append({**fields, **prediction_result})

        submissions = mapping(pd.DataFrame(predictions), competition_dir)
        if "execution_time" in submissions.columns:
            submissions = submissions.drop(columns=["execution_time"])
        if round_num is not None:
            predictions_filename = f"predictions_{round_num}.csv"
            submissions_filename = f"submissions_{round_num}.csv"
        else:
            predictions_filename = f"predictions.csv"
            submissions_filename = f"submissions.csv"
        save_csv(save_competition_dir, predictions_filename, pd.DataFrame(predictions))
        save_csv(save_competition_dir, submissions_filename, pd.DataFrame(submissions))


async def generate_and_run(
    competition_ids,
    all,
    lite,
    save_dir,
    pipeline_path,
    rounds,
    run_after=False,
    submit_after=False,
    max_concurrent: int = 1,
    data_dir: str = "tasks/mlebench"
):
    if all:
        competition_dirs = [os.path.join(data_dir, competition_id) for competition_id in registry.list_competition_ids() if competition_id != "denoising-dirty-documents"]
    else:
        competition_dirs = [
            os.path.join(data_dir, competition_id) for competition_id in competition_ids
        ]
    final_scores = []
    for competition_dir in tqdm(competition_dirs, leave=False):
        tqdm.write(f"[{Path(competition_dir).name}] Processing competition")

        # Step 1: Generate workflow for this task
        await generate_workflow_for_competition(
            competition_dir, save_dir, pipeline_path, lite, rounds, Path(competition_dir).name
        )

        if run_after:
            await run_workflow_for_competition(
                competition_dir,
                save_dir,
                lite,
                rounds,
                Path(competition_dir).name,
                max_concurrent,
            )

            if submit_after:
                report = await submit_results_for_competition(
                    competition_dir,
                    save_dir,
                    lite,
                    rounds,
                    Path(competition_dir).name,
                    data_dir,
                )
                final_scores.append(report)
    if submit_after:
        print("Average score: ", pd.DataFrame(final_scores)["avg_score"].mean())
        save_p = get_save_task_dir(data_dir, save_dir, lite)
        save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_scores))


async def run_and_submit(
    competition_ids,
    all,
    lite,
    save_dir,
    rounds,
    data_dir,
    submit_after=False,
    max_concurrent: int = 1,
):
    if all:
        competition_dirs = [os.path.join(data_dir, competition_id) for competition_id in registry.list_competition_ids() if competition_id != "denoising-dirty-documents"]
    else:
        competition_dirs = [
            os.path.join(data_dir, competition_id) for competition_id in competition_ids
        ]
    final_scores = []
    for competition_dir in tqdm(competition_dirs, leave=False):
        tqdm.write(f"[{Path(competition_dir).name}] Processing competition")

        # Step 1: Run workflow for this task
        await run_workflow_for_competition(
            competition_dir,
            save_dir,
            lite,
            rounds,
            Path(competition_dir).name,
            max_concurrent,
        )

        # Step 2: Submit results if requested
        if submit_after:
            report = await submit_results_for_competition(
                competition_dir, save_dir, lite, rounds, Path(competition_dir).name, data_dir
            )
            final_scores.append(report)
    if submit_after:
        print("Average score: ", pd.DataFrame(final_scores)["avg_score"].mean())
        save_p = get_save_task_dir(data_dir, save_dir, lite)
        save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_scores))


async def submit_results_for_competition(
    competition_dir, save_dir, lite, rounds, competition_id, data_dir
) -> dict:
    save_competition_dir = get_save_task_dir(competition_dir, save_dir, lite)
    new_registry = registry.set_data_dir(Path(data_dir))
    competition = new_registry.get_competition(competition_id)
    scores = []

    for round in range(rounds):
        round_num = round if rounds > 1 else None
        submission_path = (
            Path(save_competition_dir) / f"submissions_{round_num}.csv"
            if round_num is not None
            else Path(save_competition_dir) / "submissions.csv"
        )
        predictions_path = (
            Path(save_competition_dir) / f"predictions_{round_num}.csv"
            if round_num is not None
            else Path(save_competition_dir) / "predictions.csv"
        )
        predictions_df = pd.read_csv(predictions_path)
        report = grade_csv(submission_path, competition, lite=lite, normalize_metrics=True)
        report = report.to_dict()
        report["score"] = report["score"] if report["score"] else 0
        scores.append(
            {
                "id": (
                    f"{competition_id}_round_{round_num}"
                    if round_num is not None
                    else competition_id
                ),
                **report,
                "execution_time": predictions_df["execution_time"].mean(),
            }
        )
        save_csv(
            save_competition_dir,
            f"scores_{round_num}.csv" if round_num is not None else "scores.csv",
            pd.DataFrame(scores),
        )

    df_scores = pd.DataFrame(scores)
    avg_score = df_scores["score"].mean()
    execution_time = df_scores["execution_time"].mean()

    generation_cost_df = pd.read_csv(
        os.path.join(save_competition_dir, "generation_cost.csv")
    )
    generation_cost_df["cost"] = generation_cost_df["cost"].astype(float)
    generation_cost_df["time"] = generation_cost_df["time"].astype(float)
    avg_generation_cost = generation_cost_df["cost"].mean()
    avg_generation_time = generation_cost_df["time"].mean()
    print("Average score: ", avg_score)
    return {
        "id": competition_id,
        "avg_score": avg_score,
        "execution_time": execution_time,
        "generation_cost": avg_generation_cost,
        "generation_time": avg_generation_time,
    }


async def submit_results(competition_ids, all, lite, save_dir, rounds, data_dir):
    
    if all:
        competition_dirs = [os.path.join(data_dir, competition_id) for competition_id in registry.list_competition_ids() if competition_id != "denoising-dirty-documents"]
    else:
        competition_dirs = [
            os.path.join(data_dir, competition_id) for competition_id in competition_ids
        ]
    final_scores = []
    for competition_dir in tqdm(competition_dirs, leave=False):
        tqdm.write(f"[{Path(competition_dir).name}] Submitting results")
        report = await submit_results_for_competition(
            competition_dir, save_dir, lite, rounds, Path(competition_dir).name, data_dir
        )
        final_scores.append(report)
    save_csv(
        get_save_task_dir(data_dir, save_dir, lite),
        "final_scores.csv",
        pd.DataFrame(final_scores),
    )
