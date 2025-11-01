import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def extract_topics(metadata):
    # Đọc trực tiếp topic từ metadata
    if "topic" in metadata:
        # Kiểm tra nếu topic là list
        if isinstance(metadata["topic"], list):
            return metadata["topic"]
        # Nếu topic là string
        return [metadata["topic"]]
    return []


def analyze_task_topics():
    # Đọc metadata từ folder tasks
    tasks_path = "tasks"
    topic_by_level = {"node-level": [], "chain-level": [], "graph-level": []}

    # Đọc metadata từ mỗi task folder và subfolder
    for root, dirs, files in os.walk(tasks_path):
        if "metadata.json" in files:
            # Xác định level từ đường dẫn
            level = None
            if "node-level" in root:
                level = "node-level"
            elif "chain-level" in root:
                level = "chain-level"
            elif "graph-level" in root:
                level = "graph-level"

            if level:
                metadata_path = os.path.join(root, "metadata.json")
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        topics = extract_topics(metadata)
                        topic_by_level[level].extend(topics)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue

    # Tính tổng số topic cho mỗi level và tổng thể
    total_counts = {}
    for level, topics in topic_by_level.items():
        topic_counts = Counter(topics)
        total_counts[level] = topic_counts

    # Tạo figure với 4 subplots (3 cho mỗi level và 1 cho tổng)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Distribution of Topics in Tasks by Level", fontsize=16, y=0.95)

    # Flatten axes array để dễ dàng lặp qua
    axes = axes.flatten()

    # Vẽ biểu đồ cho từng level
    for idx, (level, counts) in enumerate(total_counts.items()):
        if counts:  # Chỉ vẽ nếu có dữ liệu
            df = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])
            df = df.sort_values("count", ascending=True)

            colors = plt.cm.Pastel1(np.linspace(0, 1, len(df)))
            ax = df.plot(kind="barh", color=colors, ax=axes[idx])
            axes[idx].set_title(f"{level} Topics", fontsize=12, pad=10)
            axes[idx].set_xlabel("Number of Tasks", fontsize=10)
            axes[idx].set_ylabel("Topics", fontsize=10)

            # Thêm giá trị lên các thanh
            for i, v in enumerate(df["count"]):
                ax.text(v, i, f" {v}", va="center", fontsize=8)

    # Vẽ biểu đồ tổng hợp ở subplot cuối
    all_topics = []
    for topics in topic_by_level.values():
        all_topics.extend(topics)

    total_topic_counts = Counter(all_topics)
    df_total = pd.DataFrame.from_dict(
        total_topic_counts, orient="index", columns=["count"]
    )
    df_total = df_total.sort_values("count", ascending=True)

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(df_total)))
    ax = df_total.plot(kind="barh", color=colors, ax=axes[3])
    axes[3].set_title("Total Topics Distribution", fontsize=12, pad=10)
    axes[3].set_xlabel("Number of Tasks", fontsize=10)
    axes[3].set_ylabel("Topics", fontsize=10)

    # Thêm giá trị lên các thanh cho biểu đồ tổng
    for i, v in enumerate(df_total["count"]):
        axes[3].text(v, i, f" {v}", va="center", fontsize=8)

    # Thêm giá trị lên các thanh
    for i, v in enumerate(df["count"]):
        ax.text(v, i, f" {v}", va="center", fontsize=10)

    plt.tight_layout()

    # Lưu biểu đồ cột ngang
    plt.savefig("results/topic_distribution.png", bbox_inches="tight", dpi=300)

    # Bổ sung: Hàm để vẽ biểu đồ hình tròn
    def plot_pie_chart(counts, title, filename):
        if counts:  # Chỉ vẽ nếu có dữ liệu
            df = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])
            labels = df.index
            sizes = df["count"]
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  # Đảm bảo biểu đồ hình tròn
            plt.title(title, fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join("results", filename), bbox_inches="tight", dpi=300)
            plt.close()

    # Bổ sung: Vẽ biểu đồ hình tròn cho từng level và tổng hợp
    for level, counts in total_counts.items():
        plot_pie_chart(counts, f"{level} Topics Distribution", f"{level}_pie.png")

    plot_pie_chart(total_topic_counts, "Total Topics Distribution", "total_pie.png")

    # In thống kê cho từng level
    for level, counts in total_counts.items():
        task_count = sum(counts.values())
        print(f"\n{level} tasks analyzed: {task_count}")
        print(f"{level} topic distribution:")
        for topic, count in counts.items():
            print(f"  {topic}: {count} tasks")

    # In thống kê tổng
    total_task_count = sum(total_topic_counts.values())
    print(f"\nTotal tasks analyzed: {total_task_count}")
    print("Overall topic distribution:")
    for topic, count in total_topic_counts.items():
        print(f"  {topic}: {count} tasks")


if __name__ == "__main__":
    analyze_task_topics()