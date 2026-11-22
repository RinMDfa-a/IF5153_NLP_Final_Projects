import json
import os
from collections import defaultdict

import pandas as pd
import requests

class PseudoSummaryDataPrepper:
    def __init__(self, url: str, data_path: str) -> None:
        self.url = url
        self.data_path = data_path
        self.generation_result = None

    def generate(self) -> None:
        with open(self.data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        results = defaultdict(dict)

        for key in data:
            if len(data[key]["reviews"]) < 3:
                continue

            Aspects = [f"{aspect}" for aspect in data[key]["raw_prediction"]]
            Reviews = [f"{review}" for review in data[key]["reviews"]]
           
            prompt = {
               "messages": [
                   {"role": "system","content": "Paraphrase, synthesize, or aggregate the following reviews into a summary at most consisting of 3 sentences without copying the reviews and includes the game name."},
                   {"role": "user", "content": f"Summarize {data[key]['app_name']} based on \n Target Aspects:{Aspects} \n Reviews:{Reviews}"}
               ]
            } 

            formatted_reviews = "\n".join(f"- {r.strip()}" for r in Reviews)

            clean_aspects = [a.replace("Aspects:", "").strip() for a in Aspects]
            formatted_aspects = ", ".join(clean_aspects)

            answer = requests.post(self.url, json=prompt)
            
            print(f"=== {key} ===")
            print(answer.json()["choices"][0]["message"]["content"])
            print("\nSummarize based on Reviews:")
            print(formatted_reviews)
            print(f"\nAspects: {formatted_aspects}\n")

            results[key]["reviews"] = (
                f"Summarize {data[key]['app_name']} based on Reviews:\n{formatted_reviews}\n\n"
                f"Aspects: {formatted_aspects}"
            )
            results[key]["summary"] = answer.json()["choices"][0]["message"]["content"]
            results[key]["ground_factual"] = formatted_reviews

        self.generation_result = results

    def save(self, save_path):
        if self.generation_result is None:
            return "Run generate() first."

        if not os.path.exists("../../data"):
            os.makedirs(save_path, exist_ok=True)
        df = pd.DataFrame.from_dict({
            "summary_instruction": [self.generation_result[key]["reviews"] for key in self.generation_result],
            "target": [self.generation_result[key]["summary"] for key in self.generation_result],
            "ground": [self.generation_result[key]["ground_factual"] for key in self.generation_result]
        })
        df.to_csv(f"../../data/phase3/{save_path}", index=False)


if __name__ == "__main__":
    # test_generator = PseudoSummaryDataPrepper(url="http://localhost:8080/v1/chat/completions", data_path="../../data/phase3/keywords_predicted.jsonl")
    # test_generator.generate()
    # test_generator.save("phase3dataset.csv")
    print(len(pd.read_csv("../../data/phase3/phase3dataset.csv")))
