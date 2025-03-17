import json
import pandas as pd
from collections import defaultdict

def excel_to_json(file_path: str) -> None:
    """
    Function convert data excel format to json format
    """

    # file_path = "dataset/data.xlsx"
    output_path = "dataset/raw_data.json"

    excel_data = pd.read_excel(file_path, sheet_name="aicore_ground_truth")

    columns_needed = ["id", "type", "text", "labeled_intent"]
    filtered_data = excel_data[columns_needed]

    filtered_data = filtered_data.dropna(subset=["labeled_intent"])
    json_data = filtered_data.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được chuyển đổi và lưu tại: {output_path}")

def get_intent_level1(input_path: str) -> None:
    """
    Function get only intent level 1
    """

    # input_path = "dataset/raw_data.json"
    output_path = "dataset/output_intent_lv1.json"

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        if "labeled_intent" in item and item["labeled_intent"]:
            item["labeled_intent"] = item["labeled_intent"].split("|")[0]

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lọc và lưu tại: {output_path}")

def dedup_data(input_path: str) -> None:
    """
    Function check duplicate data
    """

    # input_path = "dataset/output_intent_lv1.json"
    output_path = "dataset/output_dedup.json"

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    merged_data = defaultdict(lambda: {"id": None, "type": None, "text": None, "labeled_intent": []})

    for item in data:
        key = (item["id"], item["type"], item["text"])
        if merged_data[key]["id"] is None:
            merged_data[key]["id"] = item["id"]
            merged_data[key]["type"] = item["type"]
            merged_data[key]["text"] = item["text"]
        merged_data[key]["labeled_intent"].append(item["labeled_intent"])


    merged_list = [
        {
            "id": value["id"],
            "type": value["type"],
            "text": value["text"],
            "labeled_intent": list(set(value["labeled_intent"])),
        }
        for value in merged_data.values()
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(merged_list, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được gộp và lưu tại: {output_path}")


if __name__ == "__main__":
    input_file = "dataset/output.json"
    