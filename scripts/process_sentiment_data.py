"""
Process and combine ChatGPT-generated sentiment data for training.
"""

import json
import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

# Define paths
BASE_DIR = "/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot"
RAW_DIR = os.path.join(BASE_DIR, "data/sentiment/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/sentiment/processed")
FINAL_DIR = os.path.join(BASE_DIR, "data/sentiment/final")

# Create directories if they don't exist
for directory in [RAW_DIR, PROCESSED_DIR, FINAL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_raw_data():
    """Load all raw data files and combine them."""
    all_examples = []
    
    # Load standard examples
    for sentiment in ["urgent_negative", "standard_negative", "neutral", "positive"]:
        file_path = os.path.join(RAW_DIR, f"{sentiment}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    examples = json.load(f)
                print(f"Loaded {len(examples)} examples from {file_path}")
                all_examples.extend(examples)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load seed examples
    seed_path = os.path.join(RAW_DIR, "seed_examples.json")
    if os.path.exists(seed_path):
        try:
            with open(seed_path, 'r') as f:
                examples = json.load(f)
            print(f"Loaded {len(examples)} seed examples")
            all_examples.extend(examples)
        except Exception as e:
            print(f"Error loading seed examples: {e}")
    
    # Load edge cases - need special handling
    edge_path = os.path.join(RAW_DIR, "edge_cases.json")
    if os.path.exists(edge_path):
        try:
            with open(edge_path, 'r') as f:
                edge_cases = json.load(f)
            
            # Convert edge cases to standard format
            for case in edge_cases:
                all_examples.append({
                    "text": case["text"],
                    "sentiment": case["primary_sentiment"],
                    "is_edge_case": True,
                    "secondary_sentiment": case.get("secondary_sentiment")
                })
            print(f"Loaded {len(edge_cases)} edge cases")
        except Exception as e:
            print(f"Error loading edge cases: {e}")
    
    # Load conversational examples - need special handling
    conv_path = os.path.join(RAW_DIR, "conversational_examples.json")
    if os.path.exists(conv_path):
        try:
            with open(conv_path, 'r') as f:
                conversations = json.load(f)
            
            # Extract customer messages with sentiment
            conv_examples = []
            for conv in conversations:
                for turn in conv.get("turns", []):
                    if turn.get("speaker") == "customer" and "sentiment" in turn:
                        conv_examples.append({
                            "text": turn["text"],
                            "sentiment": turn["sentiment"],
                            "conversation_id": conv.get("conversation_id"),
                            "pattern": conv.get("pattern")
                        })
            
            all_examples.extend(conv_examples)
            print(f"Loaded {len(conv_examples)} conversational examples")
        except Exception as e:
            print(f"Error loading conversational examples: {e}")
    
    return all_examples

def clean_and_normalize(examples):
    """Clean and normalize the dataset."""
    cleaned = []
    
    for ex in examples:
        # Skip empty examples
        if not ex.get("text") or not ex.get("sentiment"):
            continue
        
        # Normalize text
        text = ex["text"].strip()
        
        # Normalize sentiment labels
        sentiment = ex["sentiment"].strip().lower()
        if sentiment in ["urgent negative", "urgent-negative"]:
            sentiment = "urgent_negative"
        elif sentiment in ["standard negative", "standard-negative", "negative"]:
            sentiment = "standard_negative"
        elif sentiment in ["neutral"]:
            sentiment = "neutral"
        elif sentiment in ["positive"]:
            sentiment = "positive"
        else:
            print(f"Warning: Unknown sentiment label '{sentiment}' for text: '{text}'")
            continue
        
        # Create clean example
        clean_ex = {
            "text": text,
            "sentiment": sentiment
        }
        
        # Preserve additional fields if present
        for key in ["is_edge_case", "secondary_sentiment", "conversation_id", "pattern"]:
            if key in ex:
                clean_ex[key] = ex[key]
        
        cleaned.append(clean_ex)
    
    return cleaned

def balance_dataset(examples, max_per_category=None):
    """Balance the dataset across sentiment categories."""
    # Count examples per category
    sentiment_counts = Counter([ex["sentiment"] for ex in examples])
    print("Original distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count}")
    
    # Determine target count per category
    if max_per_category is None:
        max_per_category = min(sentiment_counts.values())
    
    # Balance by randomly sampling
    balanced_examples = []
    for sentiment in ["urgent_negative", "standard_negative", "neutral", "positive"]:
        category_examples = [ex for ex in examples if ex["sentiment"] == sentiment]
        
        # If we don't have enough examples, use all of them
        if len(category_examples) <= max_per_category:
            balanced_examples.extend(category_examples)
        else:
            # Otherwise sample to the target count
            balanced_examples.extend(random.sample(category_examples, max_per_category))
    
    # Shuffle the final dataset
    random.shuffle(balanced_examples)
    
    # Print new distribution
    new_counts = Counter([ex["sentiment"] for ex in balanced_examples])
    print("Balanced distribution:")
    for sentiment, count in new_counts.items():
        print(f"  {sentiment}: {count}")
    
    return balanced_examples

def split_dataset(examples, test_size=0.15, val_size=0.15):
    """Split data into train, validation, and test sets."""
    # First split: separate test set
    train_val, test = train_test_split(
        examples, 
        test_size=test_size,
        stratify=[ex["sentiment"] for ex in examples],
        random_state=42
    )
    
    # Second split: separate validation from train
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=[ex["sentiment"] for ex in train_val],
        random_state=42
    )
    
    print(f"Split dataset into {len(train)} train, {len(val)} validation, {len(test)} test examples")
    return train, val, test

def main():
    """Main processing function."""
    # Load all data
    print("Loading raw data...")
    all_examples = load_raw_data()
    
    # Clean and normalize
    print("\nCleaning and normalizing data...")
    cleaned_examples = clean_and_normalize(all_examples)
    
    # Remove duplicates
    texts = set()
    unique_examples = []
    for ex in cleaned_examples:
        if ex["text"] not in texts:
            texts.add(ex["text"])
            unique_examples.append(ex)
    
    print(f"Removed {len(cleaned_examples) - len(unique_examples)} duplicates")
    
    # Balance dataset - aim for at least 100 examples per category
    print("\nBalancing dataset...")
    target_per_category = 150  # Adjust based on your actual data volume
    balanced_examples = balance_dataset(unique_examples, target_per_category)
    
    # Save processed dataset
    processed_path = os.path.join(PROCESSED_DIR, "processed_sentiment_data.json")
    with open(processed_path, 'w') as f:
        json.dump(balanced_examples, f, indent=2)
    print(f"Saved processed dataset with {len(balanced_examples)} examples to {processed_path}")
    
    # Split dataset
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_dataset(balanced_examples)
    
    # Save final datasets
    train_path = os.path.join(FINAL_DIR, "sentiment_train.json")
    val_path = os.path.join(FINAL_DIR, "sentiment_validation.json")
    test_path = os.path.join(FINAL_DIR, "sentiment_test.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved final datasets to {FINAL_DIR}")
    
    # Final stats
    print("\nFinal dataset statistics:")
    print(f"Train: {len(train_data)} examples")
    print(f"Validation: {len(val_data)} examples")
    print(f"Test: {len(test_data)} examples")
    
    # Print sentiment distribution in train set
    train_sentiments = Counter([ex["sentiment"] for ex in train_data])
    print("\nTrain set sentiment distribution:")
    for sentiment, count in train_sentiments.items():
        print(f"  {sentiment}: {count} ({count/len(train_data)*100:.1f}%)")

if __name__ == "__main__":
    main()