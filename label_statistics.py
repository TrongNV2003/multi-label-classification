import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_labels(dataset, unique_labels, name="Dataset"):
    label_counts = Counter()
    num_labels_per_sample = []
    
    for item in dataset:
        labels = item["label_intent"]
        label_counts.update(labels)
        num_labels_per_sample.append(len(labels))
    
    print(f"\n=== Phân tích nhãn trong {name} ===")
    print(f"Tổng số mẫu: {len(dataset)}")
    print(f"Số nhãn trung bình mỗi mẫu: {sum(num_labels_per_sample) / len(num_labels_per_sample):.2f}")
    print("Số lần xuất hiện của mỗi nhãn:")
    for label in unique_labels:
        count = label_counts.get(label, 0)
        print(f"  {label}: {count} ({count / len(dataset) * 100:.2f}%)")
    
    return label_counts, num_labels_per_sample

def visualize_label_distribution(train_counts, val_counts, test_counts, unique_labels):
    train_values = [train_counts.get(label, 0) / len(train_data) * 100 for label in unique_labels]
    val_values = [val_counts.get(label, 0) / len(val_data) * 100 for label in unique_labels]

    test_values = [test_counts.get(label, 0) / len(test_data) * 100 for label in unique_labels]
    
    x = np.arange(len(unique_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, train_values, width, label='Train', color='skyblue')
    ax.bar(x, val_values, width, label='Validation', color='lightgreen')
    ax.bar(x + width, test_values, width, label='Test', color='salmon')

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Nhãn')
    ax.set_ylabel('Tỷ lệ nhãn (%)')
    ax.set_title('Phân phối nhãn trong Train, Validation và Test set')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_labels, rotation=45, ha='right')
    ax.legend()

    for i, v in enumerate(train_values):
        ax.text(i - width, v + 1, f'{v:.1f}', ha='center', va='bottom')
    for i, v in enumerate(val_values):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    for i, v in enumerate(test_values):
        ax.text(i + width, v + 1, f'{v:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]

    with open("dataset/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open("dataset/val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    with open("dataset/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_counts, train_num_labels = analyze_labels(train_data, unique_labels, "Train set")
    val_counts, val_num_labels = analyze_labels(val_data, unique_labels, "Validation set")
    test_counts, test_num_labels = analyze_labels(test_data, unique_labels, "Test set")

    visualize_label_distribution(train_counts, val_counts, test_counts, unique_labels)