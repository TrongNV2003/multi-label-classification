import json
from collections import Counter

def analyze_and_build_hierarchy(dataset, output_file):
    hierarchy = {}
    
    for item in dataset:
        labels = item["label_intent"]
        for label in labels:
            level1, level2 = label.split("|")
            if level1 not in hierarchy:
                hierarchy[level1] = Counter()
            hierarchy[level1][level2] += 1
    
    hierarchy_dict = {level1: dict(counter) for level1, counter in hierarchy.items()}
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hierarchy_dict, f, ensure_ascii=False, indent=4)
    
    print(f"\n=== Thống kê nhãn phân cấp ===")
    print(f"Tổng số mẫu: {len(dataset)}")
    print(f"Tổng số nhãn level 1: {len(hierarchy_dict)}")
    total_level2 = sum(len(level2_dict) for level2_dict in hierarchy_dict.values())
    print(f"Tổng số nhãn level 2: {total_level2}")
    print(f"Đã lưu hierarchy vào: {output_file}")
    
    return hierarchy_dict

def map_level2_to_level1(hierarchy, level2_label):
    """Hàm mapping ngược từ level 2 về level 1"""
    for level1, level2_dict in hierarchy.items():
        if level2_label in level2_dict:
            return f"{level1}|{level2_label}"
    return None

if __name__ == "__main__":
    input_file = "dataset_speech_analyse/train.json"
    output_file = "label_hierarchy.json"
    
    with open(input_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    hierarchy = analyze_and_build_hierarchy(train_data, output_file)
    