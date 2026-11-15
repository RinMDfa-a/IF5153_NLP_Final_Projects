import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,  BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

class ABSADataset(Dataset):
    def __init__(self, data_pairs, tokenizer, max_input_length=512, max_target_len=128):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_len = max_target_len
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, index):
        item = self.data_pairs[index]
        
        model_inputs = self.tokenizer(
            item["input_text"],
            max_length=self.max_input_length,
            truncation=True
        )
        
        labels = self.tokenizer(
            text_target = item["target_text"],
            max_length = self.max_target_len,
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

class ABSAT5Model:
    def __init__(self, model_name="t5-small", model_save_dir="models/t5"):
        print(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.device="cuda"
            print(f"Using device: {self.device}")
        else:
            self.device="cpu"
            print(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.model_save_dir = model_save_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def _build_prompt(self, review_text):
        return(
            "### Instruction: Extract the key aspects and their sentiments from the following game review"
            "You MUST format the output strictly as Aspects: "
            "[category:aspect:sentiment, ...]\n\n"
            f"### Input: Review: \"{review_text}\"\n\n"
            "### Response:"
        )
    
    def load_label_data(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        training_pairs = []
        for game_id, game_data in raw_data.items():
            reviews = game_data.get("reviews", [])
            target_label = game_data.get("target_label", "")
            combination = " ".join(reviews)
            
            if combination and target_label:
                training_pairs.append({
                    "input_text": self._build_prompt(combination),
                    "target_text": target_label
                })
        
        return training_pairs
    
    def train(self, input_file, epochs=3, batch_size=4, lr=5e-5):
        training_pairs = self.load_label_data(input_file)
        train_dataset = ABSADataset(training_pairs, self.tokenizer)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model
        )
        
        t5_args = Seq2SeqTrainingArguments(
            output_dir=self.model_save_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            weight_decay=0.01,
            logging_steps=10,
            save_total_limit=3,
            num_train_epochs=epochs,
            save_strategy="no",
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        t5_trainer = Seq2SeqTrainer(
            model=self.model,
            args=t5_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        print(f"T5 Training:")
        t5_trainer.train()
        
        t5_trainer.save_model(self.model_save_dir)
        self.tokenizer.save_pretrained(self.model_save_dir)
        
        print(f"Model saved to {self.model_save_dir}")
    
    def predict(self, review_text):
        self.model.eval()
        prompt = self._build_prompt(review_text)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction


class ABSABartModel:
    def __init__(self, model_name="facebook/bart-base", model_save_dir="models/bart"):
            print(f"GPU Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.device="cuda"
                print(f"Using device: {self.device}")
            else:
                self.device="cpu"
                print(f"Using device: {self.device}")
            
            self.model_name = model_name
            self.model_save_dir = model_save_dir
            
            if os.path.exists(os.path.join(model_save_dir, "config.json")):
                load_path = model_save_dir
            else:
                load_path = model_name
                
           
            self.tokenizer = BartTokenizer.from_pretrained(load_path)
            self.model = BartForConditionalGeneration.from_pretrained(load_path).to(self.device)
            
    def _build_prompt(self, review_text):
        return(
            "### Instruction: Extract the key aspects and their sentiments from the following game review"
            "You MUST format the output strictly as Aspects: "
            "[category:aspect:sentiment, ...]\n\n"
            f"### Input: Review: \"{review_text}\"\n\n"
            "### Response:"
        )
    
    def load_label_data(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        training_pairs = []
        for game_id, game_data in raw_data.items():
            reviews = game_data.get("reviews", [])
            target_label = game_data.get("target_label", "")
            combination = " ".join(reviews)
            
            if combination and target_label:
                training_pairs.append({
                    "input_text": self._build_prompt(combination),
                    "target_text": target_label
                })
        
        return training_pairs

    def train(self, input_file, epochs=3, batch_size=4, lr=5e-5):
        training_pairs = self.load_label_data(input_file)
        train_dataset = ABSADataset(training_pairs, self.tokenizer)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model
        )
        
        bart_args = Seq2SeqTrainingArguments(
            output_dir=self.model_save_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            weight_decay=0.01,
            logging_steps=10,
            save_total_limit=3,
            num_train_epochs=epochs,
            save_strategy="no", 
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        bart_trainer = Seq2SeqTrainer(
            model=self.model,
            args=bart_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        print(f"BART Training:")
        bart_trainer.train()
        
        bart_trainer.save_model(self.model_save_dir)
        self.tokenizer.save_pretrained(self.model_save_dir)
        
        print(f"Model saved to {self.model_save_dir}")
    
    def predict(self, review_text):
        self.model.eval()
        prompt = self._build_prompt(review_text)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                no_repeat_ngram_size=3, 
                repetition_penalty=1.5,
                early_stopping=True
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction