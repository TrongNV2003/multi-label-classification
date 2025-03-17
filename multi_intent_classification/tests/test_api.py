import time
import json
import requests

API_URL = "http://0.0.0.0:2206/predict"

def measure_throughput(api_url: str, data: list) -> list:
    """
    Đo throughput của API bằng cách gửi tuần tự requests.
    
    Args:
        api_url (str): URL của API.
        data (list): Dữ liệu test.
    
    Returns:
        float: Throughput (số requests mỗi giây).
    """
    start_time = time.time()
    results = []

    for item in data:
        payload = {
            "query": item["current_message"],
            "history": item.get("history", [])
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            results.append(response.json())
        else:
            print(f"Request failed with status code: {response.status_code}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_requests = len(data)

    throughput = total_requests / elapsed_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Total requests: {total_requests}")
    print(f"Throughput: {throughput:.2f} requests/second")
    return results

if __name__ == "__main__":
    test_file = "dataset/test.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = measure_throughput(API_URL, test_data)
    
    with open("api_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Results saved to api_results.json")
