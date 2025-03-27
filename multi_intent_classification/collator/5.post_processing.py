import json
import pandas as pd

with open("multi_intent_classification/dify_dataset/processed_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for item in data:
    num_labels = len(item["label_unverify"])
    
    label_intent = item["label_intent"] + [None] * (num_labels - len(item["label_intent"]))
    
    for i in range(num_labels):
        row = {
            "id": item["id"],
            "app_id": item["app_id"],
            "conversation_id": item["conversation_id"],
            "created_at": item["created_at"],
            "current_message": item["current_message"],
            "label_unverify": item["label_unverify"][i],
            "label_intent": label_intent[i] if i < len(label_intent) else None
        }
        rows.append(row)

df = pd.DataFrame(rows)

df.to_excel("output.xlsx", index=False, engine="openpyxl")
print("Đã lưu dữ liệu vào file output.xlsx")