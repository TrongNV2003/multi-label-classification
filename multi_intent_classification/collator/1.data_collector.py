import os
import json
import pandas as pd
from collections import defaultdict

def excel_to_json(file_dir: str, output_path: str, sheet_name: str) -> None:
    """
    Function convert data excel format to json format
    """
    # json_output = []
    # for file in os.listdir(file_dir):
    #     file_path = os.path.join(file_dir, file)
    #     if not file.endswith(".xlsx"):
    #         raise ValueError(f"File {file} is not an Excel file.")
        
    excel_data = pd.read_excel(file_dir, sheet_name=sheet_name, engine="openpyxl")
    print("Available columns:", excel_data.columns.tolist())

    columns_needed = ["id", "type", "labeled_text", "labeled_intention"]
    filtered_data = excel_data[columns_needed]

    filtered_data = filtered_data.dropna(subset=["labeled_text"]).dropna(subset=["labeled_intention"])
    json_data = filtered_data.to_dict(orient="records")

    # json_output.append(json_data)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được chuyển đổi và lưu tại: {output_path}")


def dedup_data(input_path: str, output_path: str) -> None:
    """
    Function check duplicate data
    """

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print("Số lượng bản ghi ban đầu:", len(data))

    merged_data = defaultdict(lambda: {"id": None, "type": None, "labeled_text": None, "labeled_intent": []})

    for item in data:
        key = (item["id"], item["type"], item["labeled_text"])
        if merged_data[key]["id"] is None:
            merged_data[key]["id"] = item["id"]
            merged_data[key]["type"] = item["type"]
            merged_data[key]["labeled_text"] = item["labeled_text"]
        merged_data[key]["labeled_intent"].append(item["labeled_intent"])

    merged_list = [
        {
            "id": value["id"],
            "type": value["type"],
            "labeled_text": value["labeled_text"],
            "labeled_intent": list(set(value["labeled_intent"])),
        }
        for value in merged_data.values()
    ]

    print("Số lượng bản ghi sau khi gộp:", len(merged_list))

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(merged_list, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được gộp và lưu tại: {output_path}")


if __name__ == "__main__":
    # input_file = "dataset_speech_analyse/raw_data/[15-01] [COD] final_intent.xlsx"
    # output_file = "dataset_speech_analyse/raw_data_1501.json"
    # sheet_name = "ground_truth"
    # excel_to_json(input_file, output_file, sheet_name)

    input_to_dedupe = "dataset_speech_analyse/raw_data_1601.json"
    output_deduped = "dataset_speech_analyse/raw_data_deduped.json"
    dedup_data(input_to_dedupe, output_deduped)
    