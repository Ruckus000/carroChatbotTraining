#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for training DistilBERT models for chatbot tasks.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score

# Import our custom CPU trainer
from custom_trainer import CPUTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_all_models(dataset_dir: str, output_dir: str) -> None:
    """
    Train all models in the pipeline.
    
    Args:
        dataset_dir: Directory containing preprocessed datasets
        output_dir: Directory to save models
    """
    logger.info("Starting model training pipeline")
    
    # Define minimum examples required for intent classifier training
    MIN_EXAMPLES_FOR_INTENT = 3
    
    # First, train the flow classifier
    train_flow_classifier(dataset_dir, output_dir)
    
    # Train the special classifiers for fallback and clarification
    train_special_classifier("fallback", dataset_dir, output_dir)
    train_special_classifier("clarification", dataset_dir, output_dir)
    
    # Get all unique flows to train intent classifiers for
    try:
        with open(os.path.join(dataset_dir, 'flow_classification_train.json'), 'r') as f:
            flow_data = json.load(f)
            
        with open(os.path.join(dataset_dir, 'intent_classification_train.json'), 'r') as f:
            intent_data = json.load(f)
        
        # Count examples per flow
        flow_counts = {}
        for item in intent_data:
            flow = item.get('flow')
            if flow:
                flow_counts[flow] = flow_counts.get(flow, 0) + 1
                
        # Train intent classifiers for each flow with sufficient examples
        unique_flows = set(item.get('label') for item in flow_data)
        for flow in unique_flows:
            if flow_counts.get(flow, 0) >= MIN_EXAMPLES_FOR_INTENT:
                train_intent_classifier(flow, dataset_dir, output_dir)
            else:
                logger.warning(f"Skipping intent classifier for {flow} flow - only {flow_counts.get(flow, 0)} examples (minimum {MIN_EXAMPLES_FOR_INTENT})")
                
    except Exception as e:
        logger.error(f"Error in flow detection: {e}")
    
    # Entity extraction is complex - we'll handle it separately
    logger.info("Entity extraction training not implemented yet")
    
def train_flow_classifier(dataset_dir: str, output_dir: str) -> None:
    """
    Train the flow classifier using DistilBERT.
    
    Args:
        dataset_dir: Directory containing datasets
        output_dir: Directory to save the model
    """
    logger.info("Training flow classifier with DistilBERT")
    
    # Load datasets
    train_file = os.path.join(dataset_dir, 'flow_classification_train.json')
    val_file = os.path.join(dataset_dir, 'flow_classification_val.json')
    test_file = os.path.join(dataset_dir, 'flow_classification_test.json')
    
    with open(train_file, 'r') as f:
        flow_train_data = json.load(f)
    
    with open(val_file, 'r') as f:
        flow_val_data = json.load(f)
    
    # Extract unique flow labels
    unique_flows = set(item['label'] for item in flow_train_data)
    logger.info(f"Found {len(unique_flows)} unique flow labels: {sorted(list(unique_flows))}")
    
    # Create label to id mapping
    label2id = {label: idx for idx, label in enumerate(sorted(list(unique_flows)))}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert to dataset format
    train_texts = [item['text'] for item in flow_train_data]
    train_labels = [label2id[item['label']] for item in flow_train_data]
    
    val_texts = [item['text'] for item in flow_val_data]
    val_labels = [label2id[item['label']] if item['label'] in label2id else 0 for item in flow_val_data]
    
    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create datasets
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(unique_flows),
        id2label=id2label,
        label2id=label2id
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./output/checkpoints/flow_classifier',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        no_cuda=True
    )
    
    # Define metrics function
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        }
    
    # Initialize trainer
    trainer = CPUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting flow classifier training")
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # Save the model
    model_dir = os.path.join(output_dir, 'flow_classifier')
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Flow classifier saved to {model_dir}")

def train_intent_classifier(flow: str, dataset_dir: str, output_dir: str):
    """
    Train an intent classifier for a specific flow using DistilBERT.
    
    Args:
        flow: Name of the flow to train the intent classifier for
        dataset_dir: Directory containing datasets
        output_dir: Directory to save the model
    """
    logger.info(f"Training intent classifier for {flow} flow")
    
    # Define paths for data
    train_file = os.path.join(dataset_dir, 'intent_classification_train.json')
    val_file = os.path.join(dataset_dir, 'intent_classification_val.json')
    
    # Load datasets
    try:
        with open(train_file, 'r') as f:
            intent_train_data = json.load(f)
        
        with open(val_file, 'r') as f:
            intent_val_data = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Error loading intent classifier datasets: {e}")
        return
    
    # Filter data for the specific flow
    flow_train_data = [x for x in intent_train_data if x.get('flow') == flow]
    logger.info(f"Found {len(flow_train_data)} training examples for {flow} flow")
    
    # Check if we have enough examples to train
    MIN_EXAMPLES = 3
    if len(flow_train_data) < MIN_EXAMPLES:
        logger.warning(f"Not enough examples ({len(flow_train_data)}) to train intent classifier for {flow} flow. Minimum required: {MIN_EXAMPLES}")
        return
    
    # Filter validation data for this flow
    flow_val_data = [x for x in intent_val_data if x.get('flow') == flow]
    
    # If no validation data for this flow, use a portion of training data
    if len(flow_val_data) < 2:
        logger.warning(f"Insufficient validation data for {flow} flow. Using a portion of training data for validation.")
        # Use 20% of training data for validation if not enough validation data
        split_idx = max(1, int(len(flow_train_data) * 0.8))
        flow_val_data = flow_train_data[split_idx:]
        flow_train_data = flow_train_data[:split_idx]
    
    # Gather all unique intent labels from both training and validation sets
    train_labels = set(item['label'] for item in flow_train_data)
    val_labels = set(item['label'] for item in flow_val_data)
    all_labels = sorted(list(train_labels.union(val_labels)))
    
    # Check for validation labels not in training set and log warning
    missing_labels = val_labels - train_labels
    if missing_labels:
        logger.warning(f"Validation set contains labels not in training set: {missing_labels}")
    
    # Create label to id mapping
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for i, label in enumerate(all_labels)}
    
    logger.info(f"Training with {len(all_labels)} unique intent labels for {flow} flow")
    
    # Convert data to expected format
    train_texts = [x['text'] for x in flow_train_data]
    train_labels = [label2id[x['label']] for x in flow_train_data]
    
    val_texts = [x['text'] for x in flow_val_data]
    val_labels = [label2id[x['label']] if x['label'] in label2id else 0 for x in flow_val_data]
    
    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create datasets
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./output/checkpoints/{flow}_intent_classifier",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        no_cuda=True
    )
    
    # Define metrics function
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        }
    
    # Initialize trainer
    trainer = CPUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    model_output_dir = os.path.join(output_dir, f"{flow}_intent_classifier")
    os.makedirs(model_output_dir, exist_ok=True)
    
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    logger.info(f"Intent classifier for {flow} flow saved to {model_output_dir}")
    return model_output_dir

def train_special_classifier(task: str, dataset_dir: str, output_dir: str) -> None:
    """
    Train a classifier for fallback or clarification detection.
    
    Args:
        task: Either 'fallback' or 'clarification'
        dataset_dir: Directory containing datasets
        output_dir: Directory to save the model
    """
    logger.info(f"Training {task} detection classifier")
    
    # Load datasets
    train_file = os.path.join(dataset_dir, f'{task}_classification_train.json')
    val_file = os.path.join(dataset_dir, f'{task}_classification_val.json')
    
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    
    logger.info(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples for {task} detection")
    
    # Extract texts and labels
    train_texts = [item['text'] for item in train_data]
    train_labels = [1 if item['label'].startswith(task) else 0 for item in train_data]
    
    val_texts = [item['text'] for item in val_data]
    val_labels = [1 if item['label'].startswith(task) else 0 for item in val_data]
    
    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create datasets
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=2,
        id2label={0: f"not_{task}", 1: task},
        label2id={f"not_{task}": 0, task: 1}
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./output/checkpoints/{task}_classifier',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        no_cuda=True
    )
    
    # Define metrics function
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        }
    
    # Initialize trainer
    trainer = CPUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info(f"Starting {task} detection classifier training")
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results for {task} detection: {eval_result}")
    
    # Save the model
    model_dir = os.path.join(output_dir, f'{task}_classifier')
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"{task} detection classifier saved to {model_dir}")

def train_entity_extractor(dataset_dir: str, output_dir: str) -> None:
    """
    Train a named entity recognition model.
    
    Args:
        dataset_dir: Directory containing the datasets
        output_dir: Directory to save the trained model
    """
    # Force CPU
    device = torch.device("cpu")
    
    # Load datasets
    try:
        train_file = os.path.join(dataset_dir, 'entity_classification_train.json')
        val_file = os.path.join(dataset_dir, 'entity_classification_val.json')
        
        # Process will be implemented in future update
        logger.info("Entity extraction training not implemented yet")
        
    except Exception as e:
        logger.error(f"Error in entity extractor training: {e}")
