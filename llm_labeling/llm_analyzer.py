"""
multiprocessing
"""
import re
import json
import time
import numpy as np
from loguru import logger
from multiprocessing import Pool
from sklearn.metrics import precision_score, recall_score, f1_score

from openai import OpenAI
from typing import Dict, List, Optional, Union, Tuple

from llm_labeling.common import Role
from llm_labeling.llm_dataloader import Dataset
from llm_labeling.config.setting import llm_config
from llm_labeling.prompts import EXTRACT_INFO_PROMPT, SYSTEM_PROMPT

class LLMAnalyzer:
    def __init__(
        self,
        prompt_template: Optional[str] = EXTRACT_INFO_PROMPT,
        num_processes: int = 4,
        hierarchy: Optional[Dict] = None,
        
    ) -> None:
        self.prompt_template = prompt_template
        self.num_processes = num_processes
        self.hierarchy = hierarchy
        self.valid_intents = [
            "INFORM_INTENT", "NEGATE_INTENT", "AFFIRM_INTENT", "INFORM", "REQUEST", "AFFIRM", "NEGATE", "SELECT",
            "REQUEST_ALTS", "THANK_YOU", "GOODBYE", "CONFIRM", "OFFER", "NOTIFY_SUCCESS", "NOTIFY_FAILURE",
            "INFORM_COUNT", "OFFER_INTENT", "REQ_MORE", "UNKNOWN"
        ]
  

    @staticmethod
    def _parse_json(text: str) -> Dict:
        pattern = r"<output>\n(.*?)</output>"
        match = re.search(pattern, text, re.DOTALL)
        try:
            if match:
                json_text = match.group(1)
                json_dict = json.loads(json_text)
            else:
                json_dict = json.loads(text)
            if not json_dict["intention"][0]["predicted_labels"]:
                json_dict["intention"][0]["predicted_labels"] = ["UNKNOWN"]
            return json_dict
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing JSON: {e}. Raw text: {text}")
            return {"intention": [{"predicted_labels": ["UNKNOWN"]}]}

    @staticmethod
    def _inject_prompt(
            text: str,
            history: Union[List[str], str],
            prompt_template: str,
    ) -> str:
        return prompt_template.format(text=text, history=history)

    def map_level2_to_level1(self, hierarchy, level2_label):
        """Hàm mapping ngược từ level 2 về level 1"""
        for level1, level2_dict in hierarchy.items():
            if level2_label in level2_dict:
                return f"{level1}|{level2_label}"
        return "UNKNOWN|UNKNOWN"

    def analyze_text_worker(self, args: Tuple[str, str, str]) -> Tuple[Dict, float]:
        text, history, prompt_template = args
        llm = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

        prompt_str = LLMAnalyzer._inject_prompt(text, history, prompt_template)
        start_time = time.time()
        response = llm.chat.completions.create(
            seed=llm_config.seed,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            model=llm_config.model,
            messages=[
                {"role": Role.SYSTEM, "content": SYSTEM_PROMPT},
                {"role": Role.USER, "content": prompt_str},
            ],
            response_format={"type": "json_object"},
        )
        latency = time.time() - start_time
        content = response.choices[0].message.content
        result = LLMAnalyzer._parse_json(content)

        pred_labels = result["intention"][0]["predicted_labels"]
        filtered_labels = [label for label in pred_labels if label in self.valid_intents]
        if not filtered_labels:
            filtered_labels = ["UNKNOWN"]
        result["intention"][0]["predicted_labels"] = filtered_labels
        logger.info(f"Processed: Predicted labels {filtered_labels}")

        return result, latency

    def evaluate_dataset(self, dataset: Dataset, output_file: str = "results.json") -> None:
        all_true_labels = []
        all_pred_labels = []
        latencies = []
        results = []

        start_time = time.time()

        inputs = [(text, history, self.prompt_template) for id, history, text, _ in dataset]

        with Pool(processes=self.num_processes) as pool:
            process_results = pool.map(self.analyze_text_worker, inputs)

        for i, (result, latency) in enumerate(process_results):
            id, history, text, true_labels = dataset[i]
            pred_labels = result["intention"][0]["predicted_labels"]
            if self.hierarchy:
                # Map level 2 labels to level 1 labels
                pred_labels = [self.map_level2_to_level1(self.hierarchy, label) for label in pred_labels]

            true_multi_hot = [1 if label in true_labels else 0 for label in self.valid_intents]
            pred_multi_hot = [1 if label in pred_labels else 0 for label in self.valid_intents]

            all_true_labels.append(true_multi_hot)
            all_pred_labels.append(pred_multi_hot)
            latencies.append(latency)

            results.append({
                "id": id,
                "text": text,
                "history": history,
                "true_labels": true_labels,
                "predicted_labels": pred_labels,
                "latency": latency
            })

        self.calculate_latency(latencies)
        all_labels = np.array(all_true_labels)
        all_preds = np.array(all_pred_labels)
        self._calculate_accuracy(results)

        metrics = {}
        for average_type in ["micro", "macro", "weighted"]:
            avg_metrics = self._calculate_metrics(all_preds, all_labels, average_type)
            metrics[average_type] = avg_metrics
            self._print_metrics(avg_metrics, average_type)

        print(f"\nEvaluation Metrics:")
        print(f"Total time: {time.time() - start_time:.2f} seconds")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


    def _calculate_metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> Dict[str, float]:
        metrics = {}
        sample_accuracy = (all_preds == all_labels).all(axis=1).mean()
        metrics["accuracy"] = float(sample_accuracy)
        metrics["precision"] = float(precision_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["recall"] = float(recall_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["f1"] = float(f1_score(all_labels, all_preds, average=average_type, zero_division=0))
        return metrics


    def _calculate_accuracy(self, results: List[Dict]) -> None:
        correct = 0
        correct_one = 0
        total = len(results)
        for item in results:
            true_set = set(item["true_labels"])
            pred_set = set(item["predicted_labels"])
            if true_set == pred_set:
                correct += 1
            if true_set & pred_set:
                correct_one += 1
        accuracy = correct / total if total > 0 else 0
        accuracy_one = correct_one / total if total > 0 else 0
        print(f"\nAccuracy (Match one): {accuracy_one * 100:.2f}%")
        print(f"Accuracy (Match all): {accuracy * 100:.2f}%")


    def _print_metrics(self, metrics: Dict[str, float], average_type: str) -> None:
        print(f"\nMetrics ({average_type}):")
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}")
        print(f"Precision: {metrics['precision'] * 100:.2f}")
        print(f"Recall: {metrics['recall'] * 100:.2f}")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}")

    def calculate_latency(self, latencies: list) -> None:
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        mean_latency = np.mean(latencies)

        print(f"\nLatency Statistics (seconds):")
        print(f"P95 Latency: {p95_latency * 1000:.2f} ms")
        print(f"P99 Latency: {p99_latency * 1000:.2f} ms")

if __name__ == "__main__":
    # hierarchy_file = "label_hierarchy.json"
    # with open(hierarchy_file, "r", encoding="utf-8") as f:
    #     hierarchy = json.load(f) 
        
    nlu = LLMAnalyzer(num_processes=8)
    test_set = Dataset("dataset_speech_analyse/test_llm.json")
    nlu.evaluate_dataset(test_set, output_file="output_llm.json")
