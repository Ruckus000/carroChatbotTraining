import json
import os
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

# Import path utilities
from utils.path_helpers import data_file_path, model_file_path, ensure_dir_exists


def load_data(filepath):
    """Load training data from JSON file."""
    try:
        # Use resolve_path for the file path
        resolved_path = data_file_path(os.path.basename(filepath)) if filepath.startswith("data/") else filepath
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} examples from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in {filepath}")
        return []


def convert_text_entities_to_bio(text, entities):
    """Convert text and entities to word tokens with BIO tags"""
    # Split text into words
    words = text.split()
    # Initialize all tags as 'O'
    tags = ["O"] * len(words)

    for entity in entities:
        # Skip if entity doesn't have the required fields
        if (
            not isinstance(entity, dict)
            or "entity" not in entity
            or "value" not in entity
        ):
            continue

        # Get entity type, sanitize it for BIO tagging (remove spaces, special chars)
        entity_type = entity["entity"].strip()
        if not entity_type:
            continue  # Skip empty entity types

        # Get entity value
        entity_value = entity["value"]

        # Skip if entity_value is not a string
        if not isinstance(entity_value, str):
            continue

        # Skip empty values
        if not entity_value.strip():
            continue

        # Find where the entity appears in the words
        try:
            entity_words = entity_value.split()
            entity_len = len(entity_words)

            # Skip if no words in entity value
            if entity_len == 0:
                continue

            for i in range(len(words) - entity_len + 1):
                # Try to match the entire phrase
                potential_match = " ".join(words[i : i + entity_len])
                if potential_match.lower() == entity_value.lower():
                    # Mark the first word as B-entity
                    tags[i] = f"B-{entity_type}"
                    # Mark subsequent words as I-entity
                    for j in range(1, entity_len):
                        tags[i + j] = f"I-{entity_type}"
                    break
        except Exception as e:
            # Just skip this entity if there are any issues
            continue

    return words, tags


def standardize_data(data):
    """Ensure all examples have the expected format."""
    standardized = []
    for example in data:
        # Check required fields
        if "text" not in example or "intent" not in example:
            print(f"Skipping example missing required fields: {example}")
            continue

        # Ensure entities is a list
        entities = example.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        # Ensure each entity has entity and value keys
        valid_entities = []
        for entity in entities:
            if isinstance(entity, dict) and "entity" in entity and "value" in entity:
                valid_entities.append(
                    {"entity": entity["entity"], "value": entity["value"]}
                )

        standardized.append(
            {
                "text": example["text"],
                "intent": example["intent"],
                "entities": valid_entities,
            }
        )

    return standardized


def compute_intent_metrics(eval_pred):
    """Compute metrics for intent classification"""
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
        label_names = [id2intent.get(i, f"unknown-{i}") for i in unique_labels]

        # Print per-intent metrics
        print("\n=== Intent Classification Report ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall F1 Score (weighted): {f1:.4f}\n")

        # Calculate per-class metrics
        per_class_metrics = {}
        for i in unique_labels:
            label_name = id2intent.get(i, f"unknown-{i}")
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
            f"{'Intent':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
        )
        print("-" * 70)
        for intent, metrics in per_class_metrics.items():
            print(
                f"{intent:<30} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}"
            )

        # Print confusion pairs (most frequently confused intents)
        from sklearn.metrics import confusion_matrix

        print("\n=== Most Confused Intent Pairs ===")
        cm = confusion_matrix(labels, predictions)

        # Create a list of confused pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if (
                    i != j and cm[i, j] > 0
                ):  # Only include non-zero off-diagonal elements
                    true_intent = id2intent.get(i, f"unknown-{i}")
                    pred_intent = id2intent.get(j, f"unknown-{j}")
                    confused_pairs.append((true_intent, pred_intent, cm[i, j]))

        # Sort pairs by confusion count and print top 10
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"{'True Intent':<25} {'Predicted Intent':<25} {'Count':<10}")
        print("-" * 60)
        for true_intent, pred_intent, count in confused_pairs[
            :10
        ]:  # Show top 10 confused pairs
            print(f"{true_intent:<25} {pred_intent:<25} {count:<10}")

    except Exception as e:
        print(f"Error generating detailed intent metrics: {e}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compute_entity_metrics(eval_pred):
    """Compute metrics for entity recognition"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Create lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    # Convert ids to tag names and filter out -100 (padding tokens)
    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_seq.append(id2tag[labels[i][j]])
                pred_seq.append(id2tag[predictions[i][j]])

        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    # Calculate metrics using seqeval
    try:
        print("\n=== Entity Recognition Report ===")
        # Print the detailed classification report
        report_text = classification_report(true_labels, pred_labels)
        print(report_text)

        # Convert report to dictionary for metric extraction
        report = classification_report(true_labels, pred_labels, output_dict=True)

        # Calculate overall F1
        f1 = seqeval_f1_score(true_labels, pred_labels)

        # Show entity-specific metrics for B tags (beginning of entities)
        print("\n=== Entity Type Performance ===")
        print(
            f"{'Entity Type':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
        )
        print("-" * 60)

        # Extract and display metrics for entity types (only B- tags for simplicity)
        b_tag_metrics = {}
        for tag, metrics in report.items():
            if tag.startswith("B-"):
                entity_type = tag[2:]  # Remove 'B-' prefix
                b_tag_metrics[entity_type] = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"],
                }

        # Print entity type metrics
        for entity_type, metrics in sorted(b_tag_metrics.items()):
            print(
                f"{entity_type:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}"
            )

    except Exception as e:
        print(f"Error generating detailed entity metrics: {e}")
        f1 = 0.0  # Fallback
        report = {"micro avg": {"precision": 0, "recall": 0}}

    return {
        "f1": f1,
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
    }


def prepare_entity_dataset(examples, tokenizer, tag2id):
    """Prepare dataset for entity recognition."""
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, (text, entities) in enumerate(zip(examples["text"], examples["entities"])):
        try:
            # Convert to BIO format
            words, tags = convert_text_entities_to_bio(text, entities)

            # Tokenize each word and align labels
            word_tokens = []
            word_label_ids = []

            for word, tag in zip(words, tags):
                # Handle empty words (shouldn't happen but just in case)
                if not word:
                    continue

                # Tokenize the word into subwords
                subwords = tokenizer.tokenize(word)
                if not subwords:  # Handle cases where tokenization returns empty
                    subwords = [tokenizer.unk_token]

                # Add the subwords and their label
                word_tokens.extend(subwords)

                # Get tag ID, default to O if not found
                tag_id = tag2id.get(tag, tag2id["O"])

                # Add the tag ID for the first subword
                word_label_ids.append(tag_id)

                # Add -100 for the remaining subwords (to be ignored in loss)
                word_label_ids.extend([-100] * (len(subwords) - 1))

            # Add CLS and SEP tokens
            encoded_inputs = tokenizer.encode_plus(
                word_tokens,
                is_split_into_words=False,  # Already tokenized
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            # Adjust labels for CLS and SEP tokens
            labels = (
                [-100]
                + word_label_ids
                + [-100]
                * (len(encoded_inputs["input_ids"][0]) - len(word_label_ids) - 1)
            )
            labels = labels[: len(encoded_inputs["input_ids"][0])]  # Truncate if needed

            tokenized_inputs["input_ids"].append(encoded_inputs["input_ids"][0])
            tokenized_inputs["attention_mask"].append(
                encoded_inputs["attention_mask"][0]
            )
            tokenized_inputs["labels"].append(torch.tensor(labels))

        except Exception as e:
            print(f"Error preparing entity example {i}: {e}")
            # Skip this example
            continue

    # Ensure we have at least one example
    if len(tokenized_inputs["input_ids"]) == 0:
        raise ValueError("No valid examples after preprocessing. Check entity format.")

    # Convert lists to tensors
    tokenized_inputs["input_ids"] = torch.stack(tokenized_inputs["input_ids"])
    tokenized_inputs["attention_mask"] = torch.stack(tokenized_inputs["attention_mask"])
    tokenized_inputs["labels"] = torch.stack(tokenized_inputs["labels"])

    return tokenized_inputs


def train_intent_classifier(data, test_size=0.2, batch_size=16, num_epochs=3):
    """Train the intent classifier model."""
    print("\n=== Training Intent Classifier ===")

    # Make sure the model output directory exists
    intent_model_dir = model_file_path("intent_model")
    ensure_dir_exists(intent_model_dir)

    # Create dataset for intent classification
    X = [example["text"] for example in data]
    intents = list(set(example["intent"] for example in data))
    intents.sort()  # Sort for deterministic result

    global intent2id, id2intent
    intent2id = {intent: i for i, intent in enumerate(intents)}
    id2intent = {i: intent for intent, i in intent2id.items()}

    # Save intent mapping
    with open(model_file_path("intent_model/intent2id.json"), "w") as f:
        json.dump(intent2id, f, indent=2)

    # Map intents to IDs
    y = [intent2id[example["intent"]] for example in data]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Tokenize for intent classification
    intent_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_intent_encodings = intent_tokenizer(
        X_train, truncation=True, padding=True
    )
    val_intent_encodings = intent_tokenizer(X_test, truncation=True, padding=True)

    # Create datasets
    train_intent_dataset = Dataset.from_dict(
        {
            "input_ids": train_intent_encodings["input_ids"],
            "attention_mask": train_intent_encodings["attention_mask"],
            "labels": y_train,
        }
    )

    val_intent_dataset = Dataset.from_dict(
        {
            "input_ids": val_intent_encodings["input_ids"],
            "attention_mask": val_intent_encodings["attention_mask"],
            "labels": y_test,
        }
    )

    # Load models
    try:
        intent_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(intent2id),
            id2label=id2intent,
            label2id=intent2id,
        )
        intent_model.to(DEVICE)  # Move to detected device
    except Exception as e:
        print(f"Error loading intent model: {e}")
        exit(1)

    # Configure training arguments
    intent_training_args = TrainingArguments(
        output_dir=model_file_path("intent_model_checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,  # Adjusted for better learning rate
        weight_decay=0.01,
        logging_dir=model_file_path("intent_logs"),
        logging_steps=50,  # Log more frequently
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Removed no_cuda=True to allow device auto-detection
    )

    # Create trainer
    intent_trainer = Trainer(
        model=intent_model,
        args=intent_training_args,
        train_dataset=train_intent_dataset,
        eval_dataset=val_intent_dataset,
        compute_metrics=compute_intent_metrics,
    )

    # Train models
    print("Training intent model...")
    try:
        intent_trainer.train()
    except Exception as e:
        print(f"Error during intent model training: {e}")
        exit(1)

    # Save models
    print("Saving intent model...")
    try:
        intent_model.save_pretrained(model_file_path("intent_model"))
        intent_tokenizer.save_pretrained(model_file_path("intent_model"))
    except Exception as e:
        print(f"Error saving intent model: {e}")

    print("Intent classifier training complete!")


def train_entity_recognizer(data, test_size=0.2, batch_size=16, num_epochs=3):
    """Train named entity recognition model."""
    print("\n=== Training Entity Recognizer ===")

    # Make sure the model output directory exists
    entity_model_dir = model_file_path("entity_model")
    ensure_dir_exists(entity_model_dir)

    # Prepare BIO tagged data
    word_tokens = []
    word_tags = []
    entity_types = set()

    for example in data:
        for entity in example["entities"]:
            entity_types.add(entity["entity"])

    # Create BIO tags
    bio_tags = ["O"]
    for entity_type in sorted(entity_types):
        bio_tags.extend([f"B-{entity_type}", f"I-{entity_type}"])

    tag2id = {tag: i for i, tag in enumerate(bio_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    # Save tag mapping
    with open(model_file_path("entity_model/tag2id.json"), "w") as f:
        json.dump(tag2id, f, indent=2)

    # Create datasets for entity recognition
    entity_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    entity_tokenizer.model_max_length = 128
    entity_tokenizer.padding_side = "right"
    entity_tokenizer.truncation_side = "right"

    entity_dataset = prepare_entity_dataset(data, entity_tokenizer, tag2id)

    # Load models
    try:
        entity_model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(tag2id),
            id2label=id2tag,
            label2id=tag2id,
        )
        entity_model.to(DEVICE)  # Move to detected device
    except Exception as e:
        print(f"Error loading entity model: {e}")
        exit(1)

    # Configure training arguments
    entity_training_args = TrainingArguments(
        output_dir=model_file_path("entity_model_checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,  # Adjusted for better learning rate
        weight_decay=0.01,
        logging_dir=model_file_path("entity_logs"),
        logging_steps=50,  # Log more frequently
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Removed no_cuda=True to allow device auto-detection
    )

    # Create trainer
    entity_trainer = Trainer(
        model=entity_model,
        args=entity_training_args,
        train_dataset=entity_dataset,
        compute_metrics=compute_entity_metrics,
    )

    # Train models
    print("Training entity model...")
    try:
        entity_trainer.train()
    except Exception as e:
        print(f"Error during entity model training: {e}")
        exit(1)

    # Save models
    print("Saving entity model...")
    try:
        entity_model.save_pretrained(model_file_path("entity_model"))
        entity_tokenizer.save_pretrained(model_file_path("entity_model"))
    except Exception as e:
        print(f"Error saving entity model: {e}")

    print("Entity recognizer training complete!")


if __name__ == "__main__":
    # Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Use Metal Performance Shaders (MPS) for Apple Silicon
        DEVICE = torch.device("mps")
        print("Using MPS (Apple Silicon) device for training")
    elif torch.cuda.is_available():
        # Use CUDA for NVIDIA GPUs
        DEVICE = torch.device("cuda")
        print("Using CUDA device for training")
    else:
        # Fall back to CPU
        DEVICE = torch.device("cpu")
        print("GPU not available, using CPU for training")

    # Load and preprocess data
    print("Loading training data...")
    train_data = load_data("data/nlu_training_data.json")

    if not train_data:
        print("No training data available. Exiting.")
        exit(1)

    # Set aside validation data
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, random_state=42
    )

    print(
        f"Split data: {len(train_data)} training examples, {len(val_data)} validation examples"
    )

    # Standardize data format
    train_data = standardize_data(train_data)
    val_data = standardize_data(val_data)

    # Train intent classifier
    train_intent_classifier(train_data)

    # Train entity recognizer
    train_entity_recognizer(train_data)

    print("Training complete!")
