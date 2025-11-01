import os
import json
from enum import Enum
from typing import Callable, Optional
from templates.workflow_template import Workflow
from templates.pipeline_template import Pipeline
from utils.logs import logger
from scripts.calculator import Calculator
from scripts.calculator import MetricType
import pandas as pd
import numpy as np
import uuid


class Range(Enum):
    TASK = "task"
    NODE = "node"
    CHAIN = "chain"
    GRAPH = "graph"
    ALL = "all"
    MLEBENCH = "mlebench"

    @classmethod
    def values(cls):
        return [r.value for r in cls]


class EvaluationType(Enum):
    CALCULATE_SCORE = "calculate_score"
    GENERATE_AND_RUN_WORKFLOW = "generate_and_run_workflow"
    RUN_WORKFLOW = "run_workflow"
    GENERATE_WORKFLOW = "generate_workflow"
    ALL = "all"

    @classmethod
    def values(cls):
        return [r.value for r in cls]


class Evaluator:

    def __init__(self):
        self.calculator = Calculator()

    def load_workflow(
        self, folder_path: str, round_num: Optional[int] = None
    ) -> Workflow:
        folder_path = folder_path.replace("\\", ".").replace("/", ".")
        while folder_path.endswith("."):
            folder_path = folder_path[:-1]

        if round_num is not None:
            workflow_module_name = f"{folder_path}.workflow_{round_num}"
        else:
            workflow_module_name = f"{folder_path}.workflow"

        try:
            workflow_module = __import__(workflow_module_name, fromlist=[""])
            workflow_class = getattr(workflow_module, "Workflow")
            workflow = workflow_class()
            return workflow
        except ImportError as e:
            logger.info(f"Error loading workflow: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading workflow: {e}")
            raise

    def load_metric_type(self, task_folder: str) -> MetricType:
        if not os.path.exists(os.path.join(task_folder, "evaluation_type.json")):
            raise FileNotFoundError(f"Evaluation type file not found in {task_folder}")
        with open(
            os.path.join(task_folder, "evaluation_type.json"), "r", encoding="utf-8"
        ) as f:
            config = json.loads(f.read())
        return MetricType(config["metric"])

    def load_metadata(self, task_folder: str) -> dict:
        if not os.path.exists(os.path.join(task_folder, "metadata.json")):
            raise FileNotFoundError(f"Metadata file not found in {task_folder}")
        with open(
            os.path.join(task_folder, "metadata.json"), "r", encoding="utf-8"
        ) as f:
            metadata = json.loads(f.read())

            validations, validations_labels = self.load_validation_data(task_folder)
            metadata["validations"] = validations
            metadata["validations_labels"] = validations_labels
            return metadata

    def _get_file_extensions(self):
        """Get file extension definitions"""
        return {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'},
            'video': {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'},
            'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
        }
    
    def _load_files_in_folder(self, folder_path: str) -> dict:
        """
        Load files in a folder
        
        Args:
            folder_path: Path to folder to scan
            
        Returns:
            dict with image_paths, video_paths, audio_paths, text_data
        """
        extensions = self._get_file_extensions()
        result = {
            "image_paths": [],
            "video_paths": [],
            "audio_paths": [],
            "text_data": {}
        }
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in extensions['image']:
                    result["image_paths"].append(file_path)
                elif file_ext in extensions['video']:
                    result["video_paths"].append(file_path)
                elif file_ext in extensions['audio']:
                    result["audio_paths"].append(file_path)
                elif file == "record.json":
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            result["text_data"] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        result["text_data"] = {}
        
        return result

    def load_validation_data(self, task_folder: str) -> tuple[dict, pd.DataFrame]:
        validations_folder = os.path.join(task_folder, "validations")
        if not os.path.exists(validations_folder):
            raise FileNotFoundError(f"Validation folder not found in {task_folder}")
        validations = {}
        validations_labels_file = os.path.join(validations_folder, "labels.csv")
        if os.path.exists(validations_labels_file):
            validations_labels = pd.read_csv(validations_labels_file)
        else:
            validations_labels = pd.DataFrame({"id": [], "label": []})
        for _, dirs, _ in os.walk(validations_folder):
            for dir in dirs:
                validation_folder_path = os.path.join(validations_folder, dir)
                validations[dir] = self._load_files_in_folder(validation_folder_path)
        return validations, validations_labels

    def load_testcases(self, task_folder: str) -> dict:
        testcases_folder = os.path.join(task_folder, "testcases")
        if not os.path.exists(testcases_folder):
            raise FileNotFoundError(f"Testcases folder not found in {task_folder}")
        
        testcases = {}
        for _, dirs, _ in os.walk(testcases_folder):
            for dir in dirs:
                testcase_folder_path = os.path.join(testcases_folder, dir)
                testcases[dir] = self._load_files_in_folder(testcase_folder_path)
        
        return testcases

    async def process_task_run(self, task_folder: str, workflow) -> pd.DataFrame:
        """
        Process workflow run for a single task folder
        
        Args:
            task_folder: Path to task folder containing testcases
            workflow: Loaded workflow instance
            
        Returns:
            DataFrame with id and prediction columns
        """
        try:
            # Load testcases for this task
            testcases = self.load_testcases(task_folder)
            
            # Run workflow on each testcase
            predictions_data = []
            for testcase_name, testcase_data in testcases.items():
                try:
                    print(f"Processing testcase {testcase_name} - {task_folder}")
                    prediction = await workflow(testcase_data)
                    predictions_data.append({
                        "id": testcase_name,
                        "prediction": prediction
                    })
                except Exception as e:
                    print(f"Error running workflow on testcase {testcase_name}: {e}")
                    predictions_data.append({
                        "id": testcase_name,
                        "prediction": ""
                    })
            
            # Create and return DataFrame
            return pd.DataFrame(predictions_data)
            
        except Exception as e:
            print(f"Error processing task {task_folder}: {e}")
            return pd.DataFrame({"id": [], "prediction": []})
    
    def load_task_description(self, task_folder: str) -> str:
        if not os.path.exists(os.path.join(task_folder, "task_description.txt")):
            raise FileNotFoundError(f"Task description file not found in {task_folder}")
        with open(
            os.path.join(task_folder, "task_description.txt"), "r", encoding="utf-8"
        ) as f:
            return f.read()

    def load_predictions(
        self, folder: str, round_num: Optional[int] = None
    ) -> pd.DataFrame:
        try:
            if round_num is not None:
                filename = f"predictions_{round_num}.csv"
            else:
                filename = "predictions.csv"
            return pd.read_csv(os.path.join(folder, filename))
        except FileNotFoundError:
            raise FileNotFoundError(f"Predictions file not found in {folder}")

    def load_references(self, folder: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(folder, "labels.csv"))

    def find_folder_path(self, folder_name: str, base_dir: str = "tasks") -> str:
        levels = ["node-level", "chain-level", "graph-level"]
        for level in levels:
            candidate = os.path.join(base_dir, level, folder_name)
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(f"Task folder '{folder_name}' not found in any level.")

    def write_workflow(
        self, folder_path: str, code: str, round_num: Optional[int] = None
    ):
        if round_num is not None:
            filename = f"workflow_{round_num}.py"
        else:
            filename = "workflow.py"
        with open(os.path.join(folder_path, filename), "w", encoding="utf-8") as f:
            f.write(code)

    def save_csv(self, folder_path: str, file_name: str, df: pd.DataFrame):
        try:
            df.to_csv(os.path.join(folder_path, file_name), index=False)
        except PermissionError:
            file_name = f"{file_name}_{uuid.uuid4()}.csv"
            self.save_csv(folder_path, file_name, df)

    def iter_task_folders(
        self,
        _range: Range,
        result_folder_name: str,
        folder_name: Optional[str] = None,
        generate_workflow: Optional[bool] = False,
    ):
        """
        The generator returns (task_folder, result_folder, task_name) for each task corresponding to _range.
        """
        if generate_workflow:
            os.makedirs(os.path.join("results", result_folder_name), exist_ok=True)
        if _range == Range.TASK:
            if folder_name is None:
                raise ValueError(f"Task folder is required for range: {_range}")
            task_folder = self.find_folder_path(folder_name, "tasks")
            try:
                result_folder = self.find_folder_path(
                    folder_name, os.path.join("results", result_folder_name)
                )
            except FileNotFoundError as e:
                level = task_folder.split("\\")[-2]
                if generate_workflow:
                    result_folder = os.path.join(
                        "results", result_folder_name, level, folder_name
                    )
                    os.makedirs(result_folder, exist_ok=True)
                else:
                    raise e
            print(result_folder)
            yield (task_folder, result_folder, folder_name)
        elif _range in [Range.NODE, Range.CHAIN, Range.GRAPH, Range.MLEBENCH]:
            level = {
                Range.NODE: "node-level",
                Range.CHAIN: "chain-level",
                Range.GRAPH: "graph-level",
                Range.MLEBENCH: "mlebench",
            }[_range]
            task_directory = f"tasks/{level}"
            result_directory = f"results/{result_folder_name}/{level}"
            if generate_workflow:
                os.makedirs(result_directory, exist_ok=True)
            if not os.path.exists(result_directory):
                return
            for folder in (
                os.listdir(task_directory)
                if generate_workflow
                else os.listdir(result_directory)
            ):
                if (
                    os.path.isfile(os.path.join(task_directory, folder))
                    if generate_workflow
                    else os.path.isfile(os.path.join(result_directory, folder))
                ):
                    continue
                task_folder = os.path.join(task_directory, folder)
                result_folder = os.path.join(result_directory, folder)
                if generate_workflow and not os.path.exists(result_folder):
                    os.makedirs(result_folder, exist_ok=True)
                elif not os.path.exists(result_folder) or not os.path.isdir(
                    task_folder
                ):
                    logger.warning(f"Result folder not found for task: {folder}")
                    continue
                print(result_folder)

                yield (task_folder, result_folder, folder)
        elif _range == Range.ALL:
            task_directory = "tasks"
            result_directory = f"results/{result_folder_name}"
            if generate_workflow:
                os.makedirs(result_directory, exist_ok=True)
            if not os.path.exists(result_directory):
                return
            for level_folder in (
                os.listdir(task_directory)
                if generate_workflow
                else os.listdir(result_directory)
            ):
                if (
                    os.path.isfile(os.path.join(task_directory, level_folder))
                    if generate_workflow
                    else os.path.isfile(os.path.join(result_directory, level_folder))
                ):
                    continue
                for folder in (
                    os.listdir(os.path.join(task_directory, level_folder))
                    if generate_workflow
                    else os.listdir(os.path.join(result_directory, level_folder))
                ):
                    if os.path.isfile(
                        os.path.join(result_directory, level_folder, folder)
                    ) or os.path.isfile(
                        os.path.join(task_directory, level_folder, folder)
                    ):
                        continue
                    task_folder = os.path.join(task_directory, level_folder, folder)
                    result_folder = os.path.join(result_directory, level_folder, folder)
                    if generate_workflow and not os.path.exists(result_folder):
                        os.makedirs(result_folder, exist_ok=True)
                    elif not os.path.exists(result_folder) or not os.path.isdir(
                        task_folder
                    ):
                        logger.warning(f"Result folder not found for task: {folder}")
                        continue
                    print(result_folder)
                    yield (task_folder, result_folder, folder)
        else:
            raise ValueError("Invalid range")

    async def process_task_score(
        self, task_folder, result_folder, round_num: Optional[int] = None
    ):
        metric_type = self.load_metric_type(task_folder)
        try:
            predictions = self.load_predictions(result_folder, round_num)
        except FileNotFoundError:
            predictions = pd.DataFrame({"id": [], "prediction": []})
        references = self.load_references(task_folder)
        score, df = self.calculator(metric_type, predictions, references)

        if round_num is not None:
            filename = f"scores_{round_num}.csv"
        else:
            filename = "scores.csv"
        self.save_csv(result_folder, filename, df)
        return score, df

    def calculate_average_scores(self, result_folder: str, max_rounds: int):
        """Calculate average scores across all rounds and save to scores_avg.csv"""
        scores_files = []
        for round_num in range(max_rounds):
            scores_file = os.path.join(result_folder, f"scores_{round_num}.csv")
            if os.path.exists(scores_file):
                scores_files.append(pd.read_csv(scores_file))

        if not scores_files:
            return None

        # Calculate average across all rounds
        if len(scores_files) == 1:
            merged_df = scores_files[0].copy()
        else:
            # Get the structure from first dataframe
            merged_df = scores_files[0].copy()

            # Sum all score based on 'score' column
            for col in merged_df.columns:
                if col == "score" and pd.api.types.is_numeric_dtype(merged_df[col]):
                    col_sum = merged_df[col].copy()
                    for df in scores_files[1:]:
                        if col in df.columns:
                            col_sum += df[col]
                    # Calculate average
                    merged_df[col] = col_sum / len(scores_files)

        # Calculate overall average score based on 'score' column only
        if "score" in merged_df.columns:
            avg_score = merged_df["score"].mean()
        else:
            # Fallback to 0 if 'score' column doesn't exist
            avg_score = 0

        self.save_csv(result_folder, "scores_avg.csv", merged_df)
        return avg_score

    async def calculate_score(
        self,
        _range: Range,
        result_folder_name: str,
        folder_name: Optional[str] = None,
        max_rounds: Optional[int] = None,
        reset_scores: bool = True,
    ):
        """Calculate scores from existing prediction files"""
        scores = []
        tasks = []
        result_directory = None

        # Process each task folder
        for task_folder, result_folder, task_name in self.iter_task_folders(
            _range, result_folder_name, folder_name
        ):
            result_directory = (
                os.path.dirname(result_folder)
                if _range == Range.TASK
                else (
                    os.path.commonpath([result_folder, result_directory])
                    if result_directory
                    else result_folder
                )
            )

            if max_rounds:
                # Multi-round evaluation: check if scores files exist, if not create them
                scores_exist = (not reset_scores) and all(
                    os.path.exists(
                        os.path.join(result_folder, f"scores_{round_num}.csv")
                    )
                    for round_num in range(max_rounds)
                )

                if not scores_exist:
                    # Generate scores files from predictions files
                    for round_num in range(max_rounds):
                        try:
                            await self.process_task_score(
                                task_folder, result_folder, round_num
                            )
                        except:
                            print(
                                f"Error generating scores for round {round_num} in {result_folder}"
                            )
                            continue

                # Calculate average score from scores files
                avg_score = self.calculate_average_scores(result_folder, max_rounds)
                if avg_score is not None:
                    scores.append(avg_score)
                    tasks.append(task_name)
            else:
                # Single round evaluation
                score, _ = await self.process_task_score(task_folder, result_folder)
                if score is not None:
                    scores.append(score)
                    tasks.append(task_name)

        if not scores:
            return None

        final_scores = pd.DataFrame(
            {"id": range(len(tasks)), "task": tasks, "score": scores}
        )
        if _range == Range.TASK:
            self.save_csv(result_folder, "final_scores.csv", final_scores)
        elif result_directory:
            self.save_csv(result_directory, "final_scores.csv", final_scores)
        print(scores)
        return np.mean(scores)

    async def generate_workflow(
        self,
        _range: Range,
        result_folder_name: str,
        pipeline: Pipeline,
        folder_name: Optional[str] = None,
        run_workflow: bool = False,
        max_rounds: Optional[int] = None,
    ):

        # First, generate workflow files and optionally run them to create predictions
        for task_folder, result_folder, task_name in self.iter_task_folders(
            _range, result_folder_name, folder_name, generate_workflow=True
        ):

            # Generate and optionally run workflow for each round
            if max_rounds:
                for round_num in range(max_rounds):
                    try:
                        task_description = self.load_task_description(task_folder)
                        metadata = self.load_metadata(task_folder)
                        code = await pipeline(task_description, metadata)
                        self.write_workflow(result_folder, code, round_num)

                        if run_workflow:
                            workflow = self.load_workflow(result_folder, round_num)
                            predictions = await self.process_task_run(task_folder, workflow)
                            filename = f"predictions_{round_num}.csv"
                            self.save_csv(result_folder, filename, predictions)
                    except Exception as e:
                        print(
                            f"Error generating/running workflow round {round_num} for {task_folder}: {str(e)}"
                        )
                        if run_workflow:
                            filename = f"predictions_{round_num}.csv"
                            self.save_csv(
                                result_folder,
                                filename,
                                pd.DataFrame({"id": [], "prediction": []}),
                            )
            else:
                try:
                    task_description = self.load_task_description(task_folder)
                    metadata = self.load_metadata(task_folder)
                    code = await pipeline(task_description, metadata)
                    self.write_workflow(result_folder, code)

                    if run_workflow:
                        workflow = self.load_workflow(result_folder)
                        predictions = await self.process_task_run(task_folder, workflow)
                        self.save_csv(result_folder, "predictions.csv", predictions)
                except Exception as e:
                    print(f"Error generating/running workflow for {task_folder}: {e}")
                    if run_workflow:
                        self.save_csv(
                            result_folder,
                            "predictions.csv",
                            pd.DataFrame({"id": [], "prediction": []}),
                        )

        # If workflows were run, calculate scores using the existing calculate_score method
        if run_workflow:
            return await self.calculate_score(
                _range, result_folder_name, folder_name, max_rounds
            )
        else:
            return None

    async def run_workflow(
        self,
        _range: Range,
        result_folder_name: str,
        folder_name: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ):

        # First, run workflows to generate predictions files
        for task_folder, result_folder, task_name in self.iter_task_folders(
            _range, result_folder_name, folder_name
        ):

            # Run workflow for each round to generate predictions
            if max_rounds:
                for round_num in range(max_rounds):
                    try:
                        workflow = self.load_workflow(result_folder, round_num)
                        predictions = await self.process_task_run(task_folder, workflow)
                        filename = f"predictions_{round_num}.csv"
                        self.save_csv(result_folder, filename, predictions)
                        
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        print(
                            f"Error running workflow round {round_num} for {task_folder}: {e}"
                        )
                        # Save empty predictions file
                        filename = f"predictions_{round_num}.csv"
                        self.save_csv(
                            result_folder,
                            filename,
                            pd.DataFrame({"id": [], "prediction": []}),
                        )
            else:
                try:
                    workflow = self.load_workflow(result_folder)
                    
                    predictions = await self.process_task_run(task_folder, workflow)
                    self.save_csv(result_folder, "predictions.csv", predictions)
                    
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Error running workflow for {task_folder}: {e}")
                    self.save_csv(
                        result_folder,
                        "predictions.csv",
                        pd.DataFrame({"id": [], "prediction": []}),
                    )

        # Then calculate scores using the existing calculate_score method
        return await self.calculate_score(
            _range, result_folder_name, folder_name, max_rounds
        )

    async def evaluate(
        self,
        _range: Range,
        result_folder_name: str,
        max_rounds: Optional[int] = None,
        folder_name: Optional[str] = None,
        evaluation_type: EvaluationType = EvaluationType.CALCULATE_SCORE,
        pipeline: Optional[Pipeline] = None,
    ):

        if evaluation_type == EvaluationType.CALCULATE_SCORE:
            return await self.calculate_score(
                _range=_range,
                result_folder_name=result_folder_name,
                folder_name=folder_name,
                max_rounds=max_rounds,
            )
        elif evaluation_type == EvaluationType.GENERATE_AND_RUN_WORKFLOW:
            if not pipeline:
                raise ValueError(
                    "Pipeline is required for evaluation type: GENERATE_AND_RUN_WORKFLOW"
                )
            return await self.generate_workflow(
                _range=_range,
                result_folder_name=result_folder_name,
                pipeline=pipeline,
                folder_name=folder_name,
                run_workflow=True,
                max_rounds=max_rounds,
            )
        elif evaluation_type == EvaluationType.RUN_WORKFLOW:
            return await self.run_workflow(
                _range=_range,
                result_folder_name=result_folder_name,
                folder_name=folder_name,
                max_rounds=max_rounds,
            )
        elif evaluation_type == EvaluationType.GENERATE_WORKFLOW:
            if not pipeline:
                raise ValueError(
                    "Pipeline is required for evaluation type: GENERATE_AND_RUN_WORKFLOW"
                )
            return await self.generate_workflow(
                _range=_range,
                result_folder_name=result_folder_name,
                pipeline=pipeline,
                folder_name=folder_name,
                run_workflow=False,
                max_rounds=max_rounds,
            )
        elif evaluation_type == EvaluationType.ALL:
            if not pipeline:
                raise ValueError(
                    "Pipeline is required for evaluation type: GENERATE_AND_RUN_WORKFLOW"
                )
            return await self.generate_workflow(
                _range=_range,
                result_folder_name=result_folder_name,
                pipeline=pipeline,
                folder_name=folder_name,
                run_workflow=True,
                max_rounds=max_rounds,
            )
        else:
            raise ValueError("Invalid evaluation type")

