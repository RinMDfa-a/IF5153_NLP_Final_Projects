import os 
import http
import json
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
        result = {}
        with open(self.data_path, "r") as file:
            data = json.load(file)

        for key in data:
            Aspects = "\n".join(f"- {aspect}" for aspect in data[key]["aspects"])
            Reviews = "\n".join(f"- {review}" for review in data[key]["reviews"])
           
            prompt = {
               "messages": [
                   {"role": "system","content": "Paraphrase, synthesize, or aggregate the following reviews into a con summary at most consisting of 3 compact sentences without copying the reviews."},
                   {"role": "user", "content": f"Target Aspects:{Aspects} \n Reviews:{Reviews}"}
               ]
            } 

            # print(prompt)

            answer = requests.post(self.url, json=prompt)
            result[key] = answer.json()

        self.generation_result = result


    def save(self):
        pass


if __name__ == "__main__":
    test_generator = PseudoSummaryDataPrepper(url="http://localhost:8080/v1/chat/completions", data_path="../../data/example_phase2_data.json")
    test_generator.generate()

    print(json.dumps(test_generator.generation_result, indent=4))

