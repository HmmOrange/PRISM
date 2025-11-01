import os
import pandas as pd
import argparse


def calculate_average_metrics(base_folder: str, output_file: str):
    """
    Calculate the average time and cost for each competition in the specified folder.

    Args:
        base_folder (str): Path to the folder containing competition subfolders.
        output_file (str): Path to the output CSV file.
    """
    results = []

    for competition in os.listdir(base_folder):
        competition_path = os.path.join(base_folder, competition)
        generation_cost_file = os.path.join(competition_path, "generation_cost.csv")

        if os.path.isdir(competition_path) and os.path.isfile(generation_cost_file):
            df = pd.read_csv(generation_cost_file)

            avg_time = df["time"].mean()
            avg_cost = df["cost"].mean()

            results.append(
                {"competition": competition, "avg_time": avg_time, "avg_cost": avg_cost}
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.set_index("competition").T

    results_df.to_csv(output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average metrics for competitions."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="cot_code",
        # choices=["zeroshot_code", "cot_code", "fewshot_code"],
        help="Select the folder to process (e.g., zeroshot_code, cot_code, fewshot_code). Default is 'cot_code'.",
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    base_folder = os.path.join(project_root, "results", args.folder, "mlebench")
    output_file = os.path.join(
        project_root, "results", args.folder, "average_metrics.csv"
    )

    calculate_average_metrics(base_folder, output_file)
