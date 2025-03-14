import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
import os

# Load and Prepare Data
file_path = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\results\balanced_final_classified_results.csv"
df = pd.read_csv(file_path)

# Convert Text Labels to Numeric Labels
label_mapping = {"Positive": 2, "Neutral": 1, "Negative": 0}
df["label"] = df["Sentiment"].map(label_mapping)

# Load Tokenizer
model_name = "distilbert-base-uncased" # model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize Data
def tokenize_function(examples):
    return tokenizer(examples["Customer Statement"], padding="max_length", truncation=True)

dataset = Dataset.from_pandas(df[["Customer Statement", "label"]])
dataset = dataset.map(tokenize_function, batched=True)

# Split into Train & Validation Sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

# Load Pretrained Model
num_labels = 3  # Sentiment: Positive, Neutral, Negative
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Define Training Arguments
output_dir = r"C:\Uday\Consultancy\AXA Insurance\Axa_Gen_AI\fine_tuned_sentiment_model"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    learning_rate=5e-6,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True, 
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    push_to_hub=False
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train Model
trainer.train()

# Save Fine-Tuned Model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nFine-tuned model saved at: {output_dir}")
