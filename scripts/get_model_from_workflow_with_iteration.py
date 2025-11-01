import os
import click
import pandas as pd
import ast
import glob


def extract_model_ids_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    class InferenceVisitor(ast.NodeVisitor):
        def __init__(self):
            self.model_ids = []

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "model_inference":
                for kw in node.keywords:
                    if kw.arg == "model_id" and isinstance(kw.value, ast.Str):
                        self.model_ids.append(kw.value.s)
            self.generic_visit(node)

    visitor = InferenceVisitor()
    visitor.visit(tree)
    return visitor.model_ids if visitor.model_ids else ["No model"]


@click.command()
@click.option("--workflow_folder_path", type=str, required=True)
def get_model_from_workflow(workflow_folder_path):
    """
    Quét tất cả thư mục con trong workflow_folder_path, tìm các file workflow_*.py,
    trích xuất model_id và ghi vào CSV.
    """
    if not os.path.exists(workflow_folder_path):
        raise ValueError(
            f"❌ Workflow folder path {workflow_folder_path} does not exist."
        )

    records = []

    for folder in os.listdir(workflow_folder_path):
        folder_path = os.path.join(workflow_folder_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Tìm tất cả file workflow_*.py trong folder
        workflow_files = glob.glob(os.path.join(folder_path, "workflow_*.py"))
        if not workflow_files:
            records.append(
                {"task": folder, "iteration": "No workflow file", "model": "No model"}
            )
            continue

        for wf in workflow_files:
            iteration = os.path.splitext(os.path.basename(wf))[0]
            print(f"✅ Found workflow file: {wf}")
            model_ids = extract_model_ids_from_file(wf)
            for model_id in model_ids:
                records.append(
                    {"task": folder, "iteration": iteration, "model": model_id}
                )

    if records:
        df = pd.DataFrame(records)
        output_csv = os.path.join(workflow_folder_path, "model_in_workflow.csv")
        df.to_csv(output_csv, index=False)
        print(f"📁 Đã ghi file: {output_csv}")
    else:
        raise ValueError("❌ No workflow file found in the specified folder.")


if __name__ == "__main__":
    get_model_from_workflow()
