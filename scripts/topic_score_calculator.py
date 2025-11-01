import pandas as pd
import json
import os

# Đọc file CSV
df = pd.read_csv("scripts/total.csv")

# Dictionary chứa mapping giữa task và topic từ metadata.json
task_topic_mapping = {}

# Đọc topic từ metadata.json cho mỗi task từ tất cả các level
print("Đang đọc metadata cho các task...")
all_topics = set()
for task in df["task"]:
    found = False
    for level in ["node-level", "graph-level", "chain-level"]:
        metadata_path = os.path.join("tasks", level, task, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    topics_list = metadata.get("topic", [])
                    task_topic_mapping[task] = topics_list
                    all_topics.update(topics_list)
                    print(f"{task} ({level}): {topics_list}")
                    found = True
                    break
            except Exception as e:
                print(f"Lỗi đọc {metadata_path}: {e}")

    if not found:
        print(f"Không tìm thấy metadata cho task: {task}")

print(f"\nTổng cộng đọc được {len(task_topic_mapping)} tasks có metadata")
print(f"Các topic tìm thấy: {sorted(all_topics)}")

# Tạo dictionary topics từ tất cả topics tìm thấy
topics = {topic: [] for topic in all_topics}


# Tính điểm trung bình cho mỗi topic
def calculate_topic_scores(df, task_topic_mapping, topics):
    topic_scores = {topic: {"tasks": [], "scores": {}} for topic in topics}

    methods = [
        "human_design",
        "zeroshot",
        "fewshot",
        "cot",
        "zeroshot_code_iteration 5",
        "fewshot_code_iteration 5",
        "cot_code_iteration 5",
    ]

    for task in task_topic_mapping:
        if task in df["task"].values:
            task_topics = task_topic_mapping[task]
            task_row = df[df["task"] == task].iloc[0]

            for topic in task_topics:
                if topic in topics:
                    topic_scores[topic]["tasks"].append(task)
                    for method in methods:
                        if method not in topic_scores[topic]["scores"]:
                            topic_scores[topic]["scores"][method] = []
                        topic_scores[topic]["scores"][method].append(task_row[method])

    # Calculate averages
    for topic in topic_scores:
        for method in methods:
            scores = topic_scores[topic]["scores"].get(method, [])
            if scores:
                topic_scores[topic]["scores"][method] = sum(scores) / len(scores)
            else:
                topic_scores[topic]["scores"][method] = 0

    return topic_scores


def split_task_types(df):
    """Tách các task có nhiều type thành các dòng riêng biệt"""
    expanded_rows = []

    for _, row in df.iterrows():
        task_types = [t.strip() for t in row["type"].split(",")]

        for task_type in task_types:
            new_row = row.copy()
            new_row["type"] = task_type
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


def group_task_types(df):
    """Gộp các task types tương tự lại với nhau"""
    grouped_df = df.copy()

    # Gộp tất cả các loại classification
    classification_types = [
        "image-classification",
        "text-classification",
        "audio-classification",
        "tabular-classification",
        "zero-shot-classification",
        "video-classification",
        "token-classification",
    ]

    for classification_type in classification_types:
        grouped_df["type"] = grouped_df["type"].replace(
            classification_type, "classification"
        )

    # Gộp các loại generation
    generation_types = ["text-generation"]
    for generation_type in generation_types:
        grouped_df["type"] = grouped_df["type"].replace(generation_type, "generation")

    # Gộp các loại regression
    regression_types = ["tabular-regression"]
    for regression_type in regression_types:
        grouped_df["type"] = grouped_df["type"].replace(regression_type, "regression")

    return grouped_df


def calculate_overall_scores(df):
    methods = [
        "human_design",
        "zeroshot",
        "fewshot",
        "cot",
        "zeroshot_code_iteration 5",
        "fewshot_code_iteration 5",
        "cot_code_iteration 5",
    ]

    # Tách các multi-task types
    expanded_df = split_task_types(df)

    # Gộp các task types tương tự
    grouped_df = group_task_types(expanded_df)

    # Tính điểm trung bình theo type đã gộp
    type_scores = {}
    for task_type in grouped_df["type"].unique():
        type_df = grouped_df[grouped_df["type"] == task_type]
        type_scores[task_type] = {}
        for method in methods:
            type_scores[task_type][method] = type_df[method].mean()

    # Tính điểm trung bình tổng thể
    overall_scores = {}
    for method in methods:
        overall_scores[method] = df[method].mean()

    return type_scores, overall_scores, grouped_df


# Tính điểm và xuất kết quả
results = calculate_topic_scores(df, task_topic_mapping, topics)

# Tạo DataFrame để xuất ra CSV cho topics
topic_output_data = []
methods = [
    "human_design",
    "zeroshot",
    "fewshot",
    "cot",
    "zeroshot_code_iteration 5",
    "fewshot_code_iteration 5",
    "cot_code_iteration 5",
]

# Tạo bảng với approach làm hàng, topic làm cột
for method in methods:
    row = {"Approach": method}
    for topic in results:
        if results[topic]["tasks"]:  # Chỉ thêm topic có task
            row[topic] = results[topic]["scores"][method]
        else:
            row[topic] = 0
    topic_output_data.append(row)

topic_output_df = pd.DataFrame(topic_output_data)
topic_output_df.to_csv("topic_scores.csv", index=False)

# Tạo CSV riêng cho từng type và lưu kết quả tổng hợp
type_scores, overall_scores, expanded_df = calculate_overall_scores(df)

# Tạo DataFrame cho kết quả tổng hợp theo type
result_data = []
methods = [
    "human_design",
    "zeroshot",
    "fewshot",
    "cot",
    "zeroshot_code_iteration 5",
    "fewshot_code_iteration 5",
    "cot_code_iteration 5",
]

# Tạo bảng với approach làm hàng, type làm cột
for method in methods:
    row = {"Approach": method}
    for task_type in type_scores:
        row[task_type] = type_scores[task_type][method]
    row["Overall"] = overall_scores[method]
    result_data.append(row)

result_df = pd.DataFrame(result_data)
result_df.to_csv("result.csv", index=False)

# Tính và in điểm trung bình tổng thể
print("\nĐiểm trung bình theo từng type:")

for task_type, scores in type_scores.items():
    task_count = len(expanded_df[expanded_df["type"] == task_type])
    print(f"\n{task_type}: ({task_count} tasks)")
    for method, score in scores.items():
        print(f"  {method}: {score:.3f}")

print("\nĐiểm trung bình tổng thể cho tất cả các task:")
for method, score in overall_scores.items():
    print(f"{method}: {score:.3f}")
print()

# In ra màn hình số lượng task trong mỗi topic
print("\nSố lượng task trong mỗi topic:")
for topic in results:
    print(f"{topic}: {len(results[topic]['tasks'])} tasks")
    if results[topic]["tasks"]:
        print(f"Các task: {', '.join(results[topic]['tasks'])}")
    print()
# Tính và in điểm trung bình tổng thể
print("\nĐiểm trung bình theo từng type:")

for task_type, scores in type_scores.items():
    task_count = len(expanded_df[expanded_df["type"] == task_type])
    print(f"\n{task_type}: ({task_count} tasks)")
    for method, score in scores.items():
        print(f"  {method}: {score:.3f}")

print("\nĐiểm trung bình tổng thể cho tất cả các task:")
for method, score in overall_scores.items():
    print(f"{method}: {score:.3f}")
print()

# In ra màn hình số lượng task trong mỗi topic
print("\nSố lượng task trong mỗi topic:")
for topic in results:
    print(f"{topic}: {len(results[topic]['tasks'])} tasks")
    if results[topic]["tasks"]:
        print(f"Các task: {', '.join(results[topic]['tasks'])}")
    print()
