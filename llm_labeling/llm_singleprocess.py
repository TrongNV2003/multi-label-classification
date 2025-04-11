
"""
singleprocessing
"""

import json
import re
import time
import numpy as np
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score

from openai import OpenAI
from typing import Dict, List, Optional, Union

from llm_labeling.common import Role
from llm_labeling.config.setting import llm_config
from llm_labeling.prompts import EXTRACT_INFO_PROMPT, SYSTEM_PROMPT
from llm_labeling.llm_dataloader import Dataset

class LLMAnalyzer():
    def __init__(
        self,
        llm: Optional[OpenAI] = None,
        prompt_template: Optional[str] = EXTRACT_INFO_PROMPT,
    ) -> None:

        if llm is None:
            llm = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        self.llm = llm
        self.prompt_template = prompt_template
        
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
                return json_dict
            else:
                return json.loads(text)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing JSON: {e}. Raw text: {text}")
            return {"intention": [{"predicted_labels": ["UNKNOWN"]}]}
        
    def _inject_prompt(
            self,
            text: str,
            history: Union[List[str], str],
    ) -> str:
        prompt_str = self.prompt_template.format(
            text=text,
            history=history,
        )
        return prompt_str

    def analyze_text(
            self,
            text: str,
            history: Union[str, List[str]] = None,
            **kwargs,
    ) -> tuple[Dict, float]:
        prompt_str = self._inject_prompt(
            text=text,
            history=history,
        )
        start_time = time.time()
        response = self.llm.chat.completions.create(
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
        result = self._parse_json(content)

        pred_labels = result["intention"][0]["predicted_labels"]
        result["intention"][0]["predicted_labels"] = [label for label in pred_labels if label in self.valid_intents]
        return result, latency
    
    def evaluate_dataset(self, dataset: Dataset, output_file: str = "results.json") -> None:
        all_true_labels = []
        all_pred_labels = []
        latencies = []
        results = []

        start_time = time.time()
        for i in range(len(dataset)):
            history, text, true_labels = dataset[i]
            result, latency = self.analyze_text(text=text, history=history)
            
            pred_labels = result["intention"][0]["predicted_labels"]
            print(f"Processed {i}: Predicted labels {pred_labels} - True labels {true_labels}")
            true_multi_hot = [1 if label in true_labels else 0 for label in self.valid_intents]
            pred_multi_hot = [1 if label in pred_labels else 0 for label in self.valid_intents]

            all_true_labels.append(true_multi_hot)
            all_pred_labels.append(pred_multi_hot)
            latencies.append(latency)

            results.append({
                "text": text,
                "history": history,
                "true_labels": true_labels,
                "predicted_labels": pred_labels,
                "latency": latency
            })
        
        self.calculate_latency(latencies)
        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)

        accuracy = np.mean([np.array_equal(t, p) for t, p in zip(all_true_labels, all_pred_labels)])
        precision = precision_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)

        print(f"\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print(f"Total time: {time.time() - start_time:.2f} seconds")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")

    def calculate_latency(self, latencies: list) -> None:
        p99_latency = np.percentile(latencies, 99)
        print(f"P99 Latency: {p99_latency * 1000:.2f} ms")

if __name__ == "__main__":
    nlu = LLMAnalyzer()
    test_set = Dataset("multi_intent_classification/dataset/test_en.json")
    nlu.evaluate_dataset(test_set, output_file="evaluation_results.json")