import json
import os
from collections import defaultdict

import pandas as pd
import requests

example_data = {
  "input_text": "Paraphrase, synthesize, or aggregate the following Steam reviews into a concise paragraph at most consisting of 3 sentences. Adhere strictly to the target aspects and their sentiments.\n\nTarget Aspects: Content: vast (positive) | Progression: rewarding (positive) | UI: dated (negative)\n\nReviews:\n- There is an insane amount of bosses, biomes, and gear to craft.\n- Progression feels earned every time you take down hardmode bosses.\n- Inventory management and UI feel cluttered and old-school.",
  "target_text": "Players praise the vast amount of craftable content and rewarding progression system, though the inventory UI feels dated and cluttered."
}

class PseudoSummaryDataPrepper:
    def __init__(self, url: str, data_path: str) -> None:
        self.url = url
        self.data_path = data_path
        self.generation_result = None

    def generate(self) -> None:
        with open(self.data_path, "r") as file:
            data = json.load(file)

        results = defaultdict(dict)
        for key in data:

            Aspects = " ".join(f"{aspect}" for aspect in data[key]["aspects"])
            Reviews = " ".join(f"{review}" for review in data[key]["reviews"])
           
            prompt = {
               "messages": [
                   {"role": "system","content": "Paraphrase, synthesize, or aggregate the following reviews into a con summary at most consisting of 3 compact sentences without copying the reviews."},
                   {"role": "user", "content": f"Target Aspects:{Aspects} \n Reviews:{Reviews}"}
               ]
            } 

            # print(prompt)

            answer = requests.post(self.url, json=prompt)
            results[key]["reviews"] = f"Summarize based on Reviews: {Reviews}; Aspects: {Aspects}"
            results[key]["summary"] = answer.json()["choices"][0]["message"]["content"]
            results[key]["ground_factual"] = f"{Reviews}"

        self.generation_result = results

    def save(self, save_path):
        if not os.path.exists("../../data"):
            os.makedirs(save_path, exist_ok=True)
        df = pd.DataFrame.from_dict({
            "summary_instruction": [self.generation_result[key]["reviews"] for key in self.generation_result],
            "target": [self.generation_result[key]["summary"] for key in self.generation_result]
        })
        df.to_csv(f"../../data/{save_path}", index=False)


if __name__ == "__main__":
    test_generator = PseudoSummaryDataPrepper(url="http://localhost:8080/v1/chat/completions", data_path="../../data/example_phase2_data.json")
    test_generator.generate()
    test_generator.save("phase3dataset.csv")

    # print(json.dumps(test_generator.generation_result, indent=4))

