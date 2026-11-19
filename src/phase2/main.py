import os
import json
from src.phase2.pipeline.absa_pipeline import ABSAPipeline
from src.phase2.model.absa_model import ABSAT5Model, ABSABartModel

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # INPUT_FILE = os.path.join(BASE_DIR, "data", "phase2", "raw", "dataset.csv")
    # JSON_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "phase2", "preprocessed", "dataset_labeled.jsonl")
    
    INPUT_FILE = os.path.join(BASE_DIR, "data", "phase2", "raw", "relevant_reviews.parquet")
    JSON_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "phase2", "preprocessed", "relevant_reviews_labeled.jsonl")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "phase2", "bart")
    
    if not os.path.exists(JSON_OUTPUT_FILE):
        pipeline = ABSAPipeline(
            input_file=INPUT_FILE,
            output_file=JSON_OUTPUT_FILE
        )
        pipeline.run()
        
    # t5_model = ABSAT5Model(
    #     model_name="t5-base",
    #     model_save_dir=MODEL_SAVE_DIR
    # )
    
    # t5_model.train(JSON_OUTPUT_FILE, epochs=3, batch_size=4, lr=5e-5)
    
    # with open(JSON_OUTPUT_FILE, "r", encoding="utf-8") as f:
    #     label_data = json.load(f)
    
    # sample_game = label_data["1"]
    # sample_review = "".join(sample_game["reviews"])
    # expected_label = sample_game.get("targer_label", "")
    
    # result = t5_model.predict(sample_review)
    # print(f"Game: {sample_game['app_name']}")
    # print(f"Expected Label: {expected_label}")
    # print(f"Predicted Label: {result}")