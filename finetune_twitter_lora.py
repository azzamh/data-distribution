import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Configuration
config = {
    'model': {
        'name': 'gpt2',
        'max_length': 128
    },
    'lora': {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_modules': ['c_attn'],
        'bias': 'none',
        'task_type': 'SEQ_CLS'
    },
    'dataset': {
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'random_seed': 42,
        'min_samples_per_class': 100,
        'max_classes': 50,
        'text_column': 'text',
        'label_column': 'author_id'
    },
    'training': {
        'output_dir': './results',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'gradient_accumulation_steps': 1,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'logging_steps': 50,
        'eval_steps': 500,
        'save_steps': 500,
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'fp16': True if device == 'cuda' else False,
    },
    'misc': {
        'seed': 42,
        'logging_dir': './logs'
    }
}

# Set seed for reproducibility
set_seed(config['misc']['seed'])

print("✓ Konfigurasi loaded!")
print(f"Model: {config['model']['name']}")
print(f"LoRA rank: {config['lora']['r']}")
print(f"Epochs: {config['training']['num_train_epochs']}")
print(f"Learning rate: {config['training']['learning_rate']}")

# Load Twitter Support dataset
print("\n📥 Loading Twitter Support dataset...")
csv_file = "./twcs/twcs.csv"
df = pd.read_csv(csv_file)
print(f"Dataset loaded with {len(df)} samples")

# Keep only text and author_id columns
df = df[['text', 'author_id']].dropna()
print(f"After removing NaN: {len(df)}")

# Filter by minimum samples per class
min_samples = config['dataset']['min_samples_per_class']
author_counts = df['author_id'].value_counts()
valid_authors = author_counts[author_counts >= min_samples].index
df = df[df['author_id'].isin(valid_authors)]
print(f"After filtering (min {min_samples} samples): {len(df)}")

# Select top N authors
max_classes = config['dataset']['max_classes']
if max_classes and len(author_counts) > max_classes:
    top_authors = author_counts.head(max_classes).index
    df = df[df['author_id'].isin(top_authors)]
    print(f"After selecting top {max_classes} authors: {len(df)}")

# Create label mapping
unique_authors = sorted(df['author_id'].unique())
label2id = {author: idx for idx, author in enumerate(unique_authors)}
id2label = {idx: author for author, idx in label2id.items()}

df['label'] = df['author_id'].map(label2id)

print(f"\n✓ Preprocessing complete!")
print(f"Number of classes: {len(unique_authors)}")
print(f"Final dataset size: {len(df)}")

# Split dataset
train_df, temp_df = train_test_split(
    df,
    test_size=(config['dataset']['val_split'] + config['dataset']['test_split']),
    random_state=config['dataset']['random_seed'],
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=(config['dataset']['val_split'] / (config['dataset']['val_split'] + config['dataset']['test_split'])),
    random_state=config['dataset']['random_seed'],
    stratify=temp_df['label']
)

print(f"\nTrain: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

print("✓ Dataset split berhasil!")

# Load GPT-2 model
model_name = config['model']['name']
num_labels = len(label2id)

print(f"\n📦 Loading {model_name} model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

print(f"✓ Model loaded: {model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Configure LoRA
lora_config = LoraConfig(
    r=config['lora']['r'],
    lora_alpha=config['lora']['lora_alpha'],
    lora_dropout=config['lora']['lora_dropout'],
    target_modules=config['lora']['target_modules'],
    bias=config['lora']['bias'],
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
print("✓ LoRA applied to model!")
model.print_trainable_parameters()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenization function
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['model']['max_length'],
        padding=False
    )

print("\n🔄 Tokenizing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("✓ Tokenization complete!")

# Define compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Setup training arguments
training_args = TrainingArguments(
    output_dir=config['training']['output_dir'],
    num_train_epochs=config['training']['num_train_epochs'],
    per_device_train_batch_size=config['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    warmup_steps=config['training']['warmup_steps'],
    logging_dir=config['misc']['logging_dir'],
    logging_steps=config['training']['logging_steps'],
    eval_strategy='steps',
    eval_steps=config['training']['eval_steps'],
    save_steps=config['training']['save_steps'],
    save_total_limit=config['training']['save_total_limit'],
    load_best_model_at_end=config['training']['load_best_model_at_end'],
    metric_for_best_model=config['training']['metric_for_best_model'],
    greater_is_better=config['training']['greater_is_better'],
    fp16=config['training']['fp16'],
    report_to='none',
    seed=config['misc']['seed'],
)

print("✓ Training arguments configured!")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("✓ Trainer initialized!")
print(f"Ready to train on {len(train_dataset)} samples")

# Start training
print("\n🚀 Starting training...")
train_result = trainer.train()

print("\n✓ Training completed!")
print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
print(f"Final train loss: {train_result.metrics['train_loss']:.4f}")

# Save the model
trainer.save_model("./final_model")
print("✓ Model saved to ./final_model")

# Evaluate on test set
print("\n📊 Evaluating model on test set...")
test_results = trainer.evaluate(test_dataset)

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Precision: {test_results['eval_precision']:.4f}")
print(f"Recall: {test_results['eval_recall']:.4f}")
print(f"F1 Score: {test_results['eval_f1']:.4f}")
print(f"Loss: {test_results['eval_loss']:.4f}")

print("\n✓ Fine-tuning completed successfully!")