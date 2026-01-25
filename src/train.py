from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from ingestion import load_data
from preprocessing import process_data


MODEL_CHECKPOINT = "google/mt5-small"
OUTPUT_DIR = "./artifacts/model"

def train_model():
    df = load_data()
    dataset, tokenizer = process_data(df)
    
    print("[Training] Loading mT5 model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    print(f"[Registry] Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[Success] Pipeline finished successfully!")

if __name__ == "__main__":
    train_model()