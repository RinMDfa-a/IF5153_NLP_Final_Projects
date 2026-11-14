import os
from src.phase2.pipeline.absa_pipeline import ABSAPipeline

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    pipeline = ABSAPipeline(
        input_file=os.path.join(BASE_DIR, "data", "dataset.csv"),
        output_file=os.path.join(BASE_DIR, "data", "dataset_labeled.jsonl")
    )
    
    pipeline.run()