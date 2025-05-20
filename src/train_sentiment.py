#!/usr/bin/env python3
"""
Dedicated script for training the RoBERTa sentiment analysis model.
This script keeps sentiment training separate from the main train.py
while still leveraging the existing project infrastructure.
"""

import os
import json
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Import shared utilities from existing codebase
from utils.path_helpers import data_file_path, model_file_path, ensure_dir_exists

def load_sentiment_data():
    """Load the processed sentiment datasets."""
    datasets = {}
    
    # Load train, validation and test sets
    for split in ["train", "validation", "test"]:
        file_path = data_file_path(f"sentiment/final/sentiment_{split}.json")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            datasets[split] = data
            print(f"Loaded {len(data)} examples from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            datasets[split] = []
    
    return datasets["train"], datasets["validation"], datasets["test"]

def compute_sentiment_metrics(eval_pred):
    """Compute metrics for sentiment classification."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    # Print detailed per-class classification report
    try:
        # Get unique labels actually present in this batch
        unique_labels = sorted(set(np.concatenate([labels, predictions])))
        label_names = [id2sentiment.get(i, f"unknown-{i}") for i in unique_labels]

        # Print per-sentiment metrics
        print("\n=== Sentiment Classification Report ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall F1 Score (weighted): {f1:.4f}\n")

        # Calculate per-class metrics
        per_class_metrics = {}
        for i in unique_labels:
            label_name = id2sentiment.get(i, f"unknown-{i}")
            # Create binary arrays for this class
            y_true = (labels == i).astype(int)
            y_pred = (predictions == i).astype(int)

            # Calculate metrics
            cls_precision = precision_score(y_true, y_pred, zero_division=0)
            cls_recall = recall_score(y_true, y_pred, zero_division=0)
            cls_f1 = f1_score(y_true, y_pred, zero_division=0)
            support = sum(y_true)

            # Store results
            per_class_metrics[label_name] = {
                "precision": cls_precision,
                "recall": cls_recall,
                "f1": cls_f1,
                "support": support,
            }

        # Print per-class metrics in a table format
        print(
            f"{'Sentiment':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
        )
        print("-" * 70)
        for sentiment, metrics in per_class_metrics.items():
            print(
                f"{sentiment:<30} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}"
            )

    except Exception as e:
        print(f"Error generating detailed sentiment metrics: {e}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_sentiment_classifier(batch_size=32, num_epochs=5, use_large_model=True):
    """
    Train the sentiment classifier model using RoBERTa on Apple Silicon.
    
    Args:
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        use_large_model: Whether to use RoBERTa-large (True) or RoBERTa-base (False)
    """
    # Make sure the model output directory exists
    sentiment_model_dir = model_file_path("sentiment_model")
    ensure_dir_exists(sentiment_model_dir)

    # Load sentiment data
    print("Loading sentiment datasets...")
    train_data, val_data, test_data = load_sentiment_data()
    
    if not train_data:
        print("Error: No training data available. Make sure to run the data processing script first.")
        return False

    # Extract text and labels
    train_texts = [example["text"] for example in train_data]
    val_texts = [example["text"] for example in val_data]
    
    # Get unique sentiment labels and create mappings
    sentiment_labels = sorted(set(example["sentiment"] for example in train_data))
    print(f"Found sentiment categories: {sentiment_labels}")
    
    # Create mapping dictionaries
    global sentiment2id, id2sentiment
    sentiment2id = {sentiment: i for i, sentiment in enumerate(sentiment_labels)}
    id2sentiment = {i: sentiment for sentiment, i in sentiment2id.items()}
    
    # Save mapping
    with open(model_file_path("sentiment_model/sentiment2id.json"), "w") as f:
        json.dump(sentiment2id, f, indent=2)
    
    # Map labels to IDs
    train_labels = [sentiment2id[example["sentiment"]] for example in train_data]
    val_labels = [sentiment2id[example["sentiment"]] for example in val_data]
    
    # Set up device (MPS for Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU (this will be slow)")
    
    # Select model size
    model_name = "roberta-large" if use_large_model else "roberta-base"
    print(f"Using {model_name} as base model")
    
    # Load tokenizer
    sentiment_tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Tokenize with larger sequence length (viable on M4)
    max_length = 256
    print(f"Tokenizing training data with max_length={max_length}...")
    train_sentiment_encodings = sentiment_tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    val_sentiment_encodings = sentiment_tokenizer(
        val_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    
    # Create datasets
    print("Creating PyTorch datasets...")
    train_sentiment_dataset = Dataset.from_dict({
        "input_ids": train_sentiment_encodings["input_ids"],
        "attention_mask": train_sentiment_encodings["attention_mask"],
        "labels": train_labels,
    })
    
    val_sentiment_dataset = Dataset.from_dict({
        "input_ids": val_sentiment_encodings["input_ids"],
        "attention_mask": val_sentiment_encodings["attention_mask"],
        "labels": val_labels,
    })
    
    # Load model
    print(f"Loading {model_name} model...")
    sentiment_model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(sentiment2id),
        id2label=id2sentiment,
        label2id=sentiment2id,
    )
    sentiment_model.to(device)
    
    # Configure training with M4 optimizations - using older transformers API
    print("Setting up training configuration...")
    sentiment_training_args = TrainingArguments(
        output_dir=model_file_path("sentiment_model_checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,  # Use ratio instead of steps
        weight_decay=0.01,
        logging_dir=model_file_path("sentiment_logs"),
        logging_steps=20,  # More frequent logging with faster training
        # Add back basic save functionality without load_best_model_at_end
        save_steps=500,  # Save every 500 steps
        # Disable mixed precision as fp16 is not supported on MPS (Apple Silicon)
        fp16=False,
        fp16_full_eval=False
    )
    
    # Create and run trainer
    print("Creating trainer...")
    sentiment_trainer = Trainer(
        model=sentiment_model,
        args=sentiment_training_args,
        train_dataset=train_sentiment_dataset,
        eval_dataset=val_sentiment_dataset,  # Keep this for manual evaluation
        compute_metrics=compute_sentiment_metrics,
    )
    
    # Train model
    print("\n=== Starting sentiment model training ===")
    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
    print(f"Using {model_name} with batch size {batch_size} for {num_epochs} epochs\n")
    sentiment_trainer.train()
    
    # Evaluate on test set
    if test_data:
        print("\n=== Evaluating on test set ===")
        test_texts = [example["text"] for example in test_data]
        test_labels = [sentiment2id[example["sentiment"]] for example in test_data]
        
        test_encodings = sentiment_tokenizer(
            test_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        
        test_dataset = Dataset.from_dict({
            "input_ids": test_encodings["input_ids"],
            "attention_mask": test_encodings["attention_mask"],
            "labels": test_labels,
        })
        
        test_results = sentiment_trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
    
    # Save model
    print("\nSaving sentiment model...")
    sentiment_model.save_pretrained(model_file_path("sentiment_model"))
    sentiment_tokenizer.save_pretrained(model_file_path("sentiment_model"))
    
    print("\nSentiment classifier training complete!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a RoBERTa model for automotive sentiment classification")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--model", type=str, choices=["base", "large"], default="large", 
                       help="Model size (base=125M params, large=355M params)")
    
    args = parser.parse_args()
    
    # Train with specified parameters
    use_large = (args.model == "large")
    train_sentiment_classifier(batch_size=args.batch_size, num_epochs=args.epochs, use_large_model=use_large)