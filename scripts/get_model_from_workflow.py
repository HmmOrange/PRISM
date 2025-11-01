import os
import click
import pandas as pd
import ast


def extract_model_id_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    # Check for Git merge conflict markers
    lines = source.split("\n")
    conflict_detected = False
    for line in lines:
        stripped_line = line.strip()
        if (
            stripped_line.startswith("<<<<<<<")
            or stripped_line.startswith("=======")
            or stripped_line.startswith(">>>>>>>")
        ):
            conflict_detected = True
            break

    if conflict_detected:
        print(f"⚠️ Git merge conflict detected in {path}")
        return ["Git Conflict"]

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"⚠️ Syntax error in {path}")
        print(f"   Error: {e}")
        print(f"   Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
        return ["Syntax Error"]
    except Exception as e:
        print(f"⚠️ Unexpected error parsing {path}: {e}")
        return ["Parse Error"]

    class InferenceVisitor(ast.NodeVisitor):
        def __init__(self):
            self.model_ids = []

        def visit_Call(self, node):
            # Tìm hàm có tên 'model_inference'
            if isinstance(node.func, ast.Name) and node.func.id == "model_inference":
                for kw in node.keywords:
                    if kw.arg == "model_id" and isinstance(kw.value, ast.Str):
                        self.model_ids.append(kw.value.s)
            self.generic_visit(node)

    visitor = InferenceVisitor()
    visitor.visit(tree)
    if visitor.model_ids:
        return visitor.model_ids
    else:
        return ["No model"]


@click.command()
@click.option("--workflow_folder_path", type=str, required=True)
def get_model_from_workflow(workflow_folder_path):
    """
    Quét tất cả thư mục con trong workflow_folder_path, tìm file workflow.py,
    trích xuất model_id và ghi vào CSV.
    """
    if not os.path.exists(workflow_folder_path):
        raise ValueError(
            f"❌ Workflow folder path {workflow_folder_path} does not exist."
        )

    task_models = {}

    for folder in os.listdir(workflow_folder_path):
        folder_path = os.path.join(workflow_folder_path, folder)

        # Nếu không phải thư mục => bỏ qua
        if not os.path.isdir(folder_path):
            continue

        workflow_file = os.path.join(folder_path, "workflow.py")
        if os.path.exists(workflow_file):
            print(f"✅ Found workflow file: {workflow_file}")
            model_ids = extract_model_id_from_file(workflow_file)

            if folder not in task_models:
                task_models[folder] = []
            task_models[folder].extend(model_ids)

    if task_models:
        records = []
        for task, models in task_models.items():
            # Loại bỏ duplicate và join bằng dấu phẩy
            unique_models = list(
                dict.fromkeys(models)
            )  # Preserve order while removing duplicates
            model_string = ", ".join(unique_models)
            records.append({"task": task, "model": model_string})

        df = pd.DataFrame(records)
        output_csv = os.path.join(workflow_folder_path, "model_in_workflow.csv")
        df.to_csv(output_csv, index=False)
        print(f"📁 Đã ghi file: {output_csv}")
    else:
        raise ValueError("❌ No workflow file found in the specified folder.")


if __name__ == "__main__":
    get_model_from_workflow()
