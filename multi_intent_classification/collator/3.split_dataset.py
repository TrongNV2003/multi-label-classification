from torch.utils.data import Subset
import json
import random


def split_dataset(dataset: list, train_ratio: float = 0.7, valid_ratio: float = 0.15, test_ratio: float = 0.15):
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_set = [dataset[i] for i in train_indices]
    valid_set = [dataset[i] for i in valid_indices]
    test_set = [dataset[i] for i in test_indices]

    return train_set, valid_set, test_set

def split_data(input_path: str, save: bool = True) -> None:
    """
    Function split data into train, val, test
    """

    with open(input_path, "r", encoding="utf-8") as file:
        full_dataset = json.load(file)

    train_set, valid_set, test_set = split_dataset(
        full_dataset,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15
    )

    if save:
        with open("gen_dataset/gen_train.json", "w", encoding="utf-8") as train_file:
            json.dump(train_set, train_file, ensure_ascii=False, indent=4)

        with open("gen_dataset/gen_val.json", "w", encoding="utf-8") as val_file:
            json.dump(valid_set, val_file, ensure_ascii=False, indent=4)

        with open("gen_dataset/gen_test.json", "w", encoding="utf-8") as test_file:
            json.dump(test_set, test_file, ensure_ascii=False, indent=4)

        print("Dữ liệu đã được chia và lưu vào các file: train.json, val.json, test.json")

    # Kiểm tra phần tử trùng lặp giữa các tập
    train_ids = {item["current_message"] for item in train_set}
    val_ids = {item["current_message"] for item in valid_set}
    test_ids = {item["current_message"] for item in test_set}

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
    input_file = "gen_dataset/gen_processed.json"
    split_data(input_file)
    


