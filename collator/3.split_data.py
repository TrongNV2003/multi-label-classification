import json

def split_data(input_path: str, save: bool = True) -> None:
    """
    Function split data into train, val, test
    """

    input_path = "dataset/output.json"
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    val_data = data[:170]
    test_data = data[170:342]
    train_data = data[342:]

    if save:
        with open("train.json", "w", encoding="utf-8") as train_file:
            json.dump(train_data, train_file, ensure_ascii=False, indent=4)

        with open("val.json", "w", encoding="utf-8") as val_file:
            json.dump(val_data, val_file, ensure_ascii=False, indent=4)

        with open("test.json", "w", encoding="utf-8") as test_file:
            json.dump(test_data, test_file, ensure_ascii=False, indent=4)

        print("Dữ liệu đã được chia và lưu vào các file: train.json, val.json, test.json")

    # Kiểm tra phần tử trùng lặp giữa các tập
    train_ids = {item["current_message"] for item in train_data}
    val_ids = {item["current_message"] for item in val_data}
    test_ids = {item["current_message"] for item in test_data}

    # Kiểm tra giao của các tập
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print(train_val_overlap)
    print(train_test_overlap)
    print(val_test_overlap)

    print(f"Số phần tử trùng giữa train và val: {len(train_val_overlap)}")
    print(f"Số phần tử trùng giữa train và test: {len(train_test_overlap)}")
    print(f"Số phần tử trùng giữa val và test: {len(val_test_overlap)}")


if __name__ == "__main__":
    input_file = "dataset/output.json"
    split_data(input_file)
    