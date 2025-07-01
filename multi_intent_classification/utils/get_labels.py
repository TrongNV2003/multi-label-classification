import json

def get_unique_labels(input_file: str, label_col: str = "label") -> list:
    unique_labels = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            if isinstance(record[label_col], list):
                for label in record[label_col]:
                    unique_labels.add(label)
            else:
                unique_labels.add(record[label_col])

    outputs_labels_list = list(unique_labels)
    
    print(f"\nDanh sách nhãn: {outputs_labels_list}")
    print(f"Tổng số lượng nhãn: {len(outputs_labels_list)}")
    return outputs_labels_list