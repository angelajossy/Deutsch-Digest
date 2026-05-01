import torch
from transformers import AutoTokenizer


MODEL_CHECKPOINT = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self): return len(self.inputs["input_ids"])
    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx]
        }

def process_data(df):
   
    print("[Preprocessing] Tokenizing data...")
    
    inputs = tokenizer(
        df["text"].tolist(), max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    targets = tokenizer(
        df["summary"].tolist(), max_length=64, truncation=True, padding="max_length", return_tensors="pt"
    )
    
    dataset = SimpleDataset(inputs, targets)
    print("[Preprocessing] Data ready for training.")
    return dataset, tokenizer
