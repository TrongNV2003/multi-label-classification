# import json
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt

# def analyze_labels(dataset, unique_labels):
#     label_counts = Counter()
#     num_labels_per_sample = []
    
#     for item in dataset:
#         labels = item["label_intent"]
#         # Chỉ lấy nhãn cấp 1 (phần trước dấu "|") từ mỗi nhãn trong danh sách
#         # level1_labels = [label.split("|")[0] for label in labels]
#         label_counts.update(labels)
#         num_labels_per_sample.append(len(labels))
    
#     print(f"\n=== Phân tích nhãn Dataset ===")
#     print(f"Tổng số mẫu: {len(dataset)}")
#     print(f"Số nhãn trung bình mỗi mẫu: {sum(num_labels_per_sample) / len(num_labels_per_sample):.2f}")
#     print("Số lần xuất hiện của mỗi nhãn cấp 1:")
#     for label in unique_labels:
#         count = label_counts.get(label, 0)
#         print(f"{label}: {count} ({count / len(dataset) * 100:.2f}%)")
    
#     return label_counts, num_labels_per_sample

# def visualize_label_distribution(json_data, train_counts, unique_labels):

#     train_values = [train_counts.get(label, 0) / len(json_data) * 100 for label in unique_labels]

#     train_counts_absolute = [train_counts.get(label, 0) for label in unique_labels]
    
#     x = np.arange(len(unique_labels))
#     width = 0.25

#     fig, ax = plt.subplots(figsize=(14, 6))
#     bars = ax.bar(x - width, train_values, width, color='skyblue')

#     # Tùy chỉnh biểu đồ
#     ax.set_xlabel('Nhãn Level 1')
#     ax.set_ylabel('Tỷ lệ nhãn (%)')
#     ax.set_title('Phân phối nhãn Level 1')
#     ax.set_xticks(x)
#     ax.set_xticklabels(unique_labels, rotation=45, ha='right')

#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         count = train_counts_absolute[i]
#         ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{count}', 
#                 ha='center', va='bottom', fontsize=10)

#     plt.tight_layout()
#     plt.show()

# def get_unique_labels(input_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     outputs_labels_count = Counter()

#     for record in data:
#         # level1_labels = [label.split("|")[0] for label in record["label_intent"]]
#         outputs_labels_count.update(record["label_intent"])

#     outputs_labels_list = [label for label, _ in outputs_labels_count.most_common()]

#     print(f"\nDanh sách nhãn cấp 1 trong workflow_outputs: {outputs_labels_list}")
#     print(f"Tổng số lượng nhãn cấp 1: {len(outputs_labels_list)}")
#     return outputs_labels_list

# if __name__ == "__main__":
#     input_file = "dataset_speech_analyse/raw_data/processed_data_history_level2.json"
    
#     unique_labels = get_unique_labels(input_file)

#     with open("dataset_speech_analyse/raw_data/processed_data_history_level2.json", "r", encoding="utf-8") as f:
#         json_data = json.load(f)


#     dataset_counts, train_num_labels = analyze_labels(json_data, unique_labels)

#     visualize_label_distribution(json_data, dataset_counts, unique_labels)



"""
Thống kê nhãn level 2, vì nhiều quá nên chỉ lấy top 20 nhãn nhiều nhất
"""
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def analyze_labels(dataset, unique_labels):
    label_counts = Counter()
    num_labels_per_sample = []
    
    for item in dataset:
        labels = item["label_intent"]
        # level2_labels = [label.split("|")[1] for label in labels]  # Lấy nhãn cấp 2
        label_counts.update(labels)
        num_labels_per_sample.append(len(labels))
    
    print(f"\n=== Phân tích nhãn trong Dataset ===")
    print(f"Tổng số mẫu: {len(dataset)}")
    print(f"Số nhãn trung bình mỗi mẫu: {sum(num_labels_per_sample) / len(num_labels_per_sample):.2f}")
    print("Số lần xuất hiện của mỗi nhãn cấp 2 (chi tiết lưu vào file):")
    

    stats = {label: {"count": label_counts.get(label, 0), 
                     "percentage": label_counts.get(label, 0) / len(dataset) * 100} 
             for label in unique_labels}
    df = pd.DataFrame.from_dict(stats, orient='index').sort_values(by="count", ascending=False)
    df.to_csv(f"label_stats_dataset.csv", encoding='utf-8')
    print(f"Đã lưu thống kê chi tiết vào: label_stats_dataset.csv")
    
    return label_counts, num_labels_per_sample

def visualize_label_distribution(json_data, train_counts, top_n=20):
    # Lấy top N nhãn phổ biến nhất
    top_labels = [label for label, count in train_counts.most_common(top_n)]
    top_counts = [train_counts[label] for label in top_labels]
    top_percentages = [count / len(json_data) * 100 for count in top_counts]
    
    # Tính tổng số lượng của các nhãn còn lại (Others)
    others_count = sum(train_counts.values()) - sum(top_counts)
    if others_count > 0:
        top_labels.append("Others")
        top_counts.append(others_count)
        top_percentages.append(others_count / len(json_data) * 100)
    
    x = np.arange(len(top_labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(x, top_percentages, width, color='skyblue')

    ax.set_xlabel('Nhãn Level 2 (Top 30 + Others)')
    ax.set_ylabel('Tỷ lệ nhãn (%)')
    ax.set_title(f'Phân phối Top {top_n} nhãn level 2')
    ax.set_xticks(x)
    ax.set_xticklabels(top_labels, rotation=45, ha='right')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = top_counts[i]
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{count}', 
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def get_unique_labels(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    outputs_labels_count = Counter()

    for record in data:
        # level2_labels = [label.split("|")[1] for label in record["label_intent"]]
        outputs_labels_count.update(record["label_intent"])

    outputs_labels_list = [label for label, _ in outputs_labels_count.most_common()]

    print(f"\nDanh sách nhãn cấp 2 trong workflow_outputs: {outputs_labels_list}")
    print(f"Tổng số lượng nhãn cấp 2: {len(outputs_labels_list)}")
    return outputs_labels_list

if __name__ == "__main__":
    input_file = "dataset_speech_analyse/train.json"
    
    unique_labels = get_unique_labels(input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    train_counts, train_num_labels = analyze_labels(json_data, unique_labels)
    visualize_label_distribution(json_data, train_counts, top_n=30)