import json
import pandas as pd
from src.phase2.preprocessing.extractor import YAKEExtractor
from src.phase2.preprocessing.formatter import PromptFormatter
from src.phase2.preprocessing.ontology import OntologyMapper
import concurrent.futures
from collections import defaultdict
import os

class ABSAPipeline:
    def __init__(self, input_file, output_file, chunk_size=10000, max_reviews_per_game=8, max_games=500, min_relevance=0.4):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.max_reviews_per_game = max_reviews_per_game
        self.max_games = max_games
        self.min_relevance = min_relevance

        self.extractor = YAKEExtractor()
        self.mapper = OntologyMapper()
        self.formatter = PromptFormatter()


    def _iter_chunks(self):
        if self.input_file.endswith(".parquet"):
            df = pd.read_parquet(self.input_file)
            
            if "relevance_score" in df.columns:
                df = df[df["relevance_score"] >= self.min_relevance]
            elif "is_relevant" in df.columns:
                df = df[df["is_relevant"] == True]
            
            for start in range(0, len(df), self.chunk_size):
                yield df.iloc[start: start + self.chunk_size]
        
        else:
            yield from pd.read_csv(self.input_file, chunksize=self.chunk_size)
    
    def _process_single_review(self, row_tuple):
        app_name, review_text = row_tuple
        if not str(review_text.strip()) or str(review_text.lower()) == 'nan':
            return None
        
        keywords = self.extractor.extract(review_text)
        
        aspects_list = []
        task_labels = []
        
        for key_word, _ in keywords:
            ignore_words = {"game", "games", "play", "playing", "steam"}
            if key_word.lower() in ignore_words or key_word.lower() in app_name.lower():
                continue
            
            
            category = self.mapper.get_category(key_word)
            opinion = self.mapper.get_opinion(review_text)
            sentiment = self.mapper.get_sentiment(review_text, key_word)
            aspects_list.append({
                "category": category,
                "aspect": key_word,
                "opinion": opinion,
                "sentiment": sentiment,
            })
            
            task_labels.append(f"{category}:{key_word.lower()}:{sentiment.lower()}")
            task_labels = sorted(set(task_labels))
            
        if not task_labels:
            return None
            
        return {
            "app_name": app_name,
            "review_text": review_text,
            "aspects": aspects_list,
            "task_labels": task_labels
        }
            
    def run(self):
        max_workers = max(1, os.cpu_count() - 1)
        game_groups = defaultdict(lambda: {"reviews": [], "aspects": [], "labels": [], "review_pairs" : []})
        query_count = defaultdict(int)
        
        for chunk_id, chunk in enumerate(self._iter_chunks()):
            if 'app_name' in chunk.columns:
                group_col = 'app_name'
            elif 'app_id' in chunk.columns:
                group_col = 'app_id' 
            
            chunk = chunk.dropna(subset=['review_text', group_col])
            chunk = chunk.groupby(group_col).head(self.max_reviews_per_game)
            
            row_to_process = []
            for app_name, review_text in zip(chunk[group_col], chunk['review_text']):
                game = str(app_name)
                
                if game not in query_count and len(query_count) >= self.max_games:
                    continue
                
                if query_count[game] < self.max_reviews_per_game:
                    row_to_process.append((game, review_text))
                    query_count[game] += 1
            
            if not row_to_process:
                print(f"Batch: {chunk_id + 1} skipped (games in this chunk already full).")
                continue
        
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(self._process_single_review, row_to_process)
                
            for item in results:
                if item:
                    game = item["app_name"]
                    if len(game_groups[game]["reviews"]) < self.max_reviews_per_game:
                        game_groups[game]["reviews"].append(item["review_text"])
                        game_groups[game]["aspects"].extend(item["aspects"])
                        game_groups[game]["labels"] = sorted(set(game_groups[game]["labels"] + item["task_labels"]))
                        game_groups[game]["review_pairs"].append({
                            "review": item["review_text"],
                            "target_label": f"Aspects: [{', '.join(item['task_labels'])}]"
                        })
                        
            output = {}
        
            for index, (app_name, data) in enumerate(game_groups.items(), start=1):
                output[str(index)] = self.formatter.format_merge_game(
                    app_name = app_name,
                    reviews_list = data["reviews"],
                    aspects_list = data["aspects"],
                    seq2seq_labels = data["labels"],
                    review_pairs = data["review_pairs"]
                )
                
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', encoding='utf-8') as file:
                json.dump(output, file, indent=4, ensure_ascii=False)
                
            print(f"Output saved to {self.output_file}")
        
            print(f"Batch: {chunk_id + 1} processed.({len(game_groups)}/{self.max_games} games)")
            
            
       