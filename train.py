import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def main():
    print("--- TG-LLM RoBERTa Fine-Tuning ---")
    model_name = "distilroberta-base"
    
    # 1. Load Tokenizer and Model
    # num_labels=1 replaces the word-generation head with a continuous float node
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # 2. Load the Dataset
    df = pd.read_csv("telemetry_data.csv")
    df['label'] = df['label'].astype(float) # PyTorch requires strict float types for MSE Loss
    dataset = Dataset.from_pandas(df)

    # 3. Tokenize the text (translating words to numerical IDs)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 4. Split Data (80% Training / 20% Validation)
    split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # 5. Define Training Parameters
    training_args = TrainingArguments(
        output_dir="./tg_roberta_checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
    )

    # 6. Initialize and Execute
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Initiating fine-tuning (MSE Loss Optimization)...")
    trainer.train()

    # 7. Save the optimized model
    model.save_pretrained("./tg-roberta-finetuned")
    tokenizer.save_pretrained("./tg-roberta-finetuned")
    print("Training complete! Model saved to ./tg-roberta-finetuned")

if __name__ == "__main__":
    main()