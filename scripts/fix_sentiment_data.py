"""
Quick fix script to rebuild sentiment datasets with all categories.
This script ensures all four sentiment categories are included in the final datasets.
"""

import json
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# Define paths
BASE_DIR = "/Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot"
RAW_DIR = os.path.join(BASE_DIR, "data/sentiment/raw")
FINAL_DIR = os.path.join(BASE_DIR, "data/sentiment/final")

# Make sure final directory exists
if not os.path.exists(FINAL_DIR):
    os.makedirs(FINAL_DIR)

def load_all_raw_data():
    """Load all raw sentiment data files."""
    data = []
    
    # Load standard category files
    for sentiment in ["urgent_negative", "standard_negative", "neutral", "positive"]:
        file_path = os.path.join(RAW_DIR, f"{sentiment}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    examples = json.load(f)
                print(f"Loaded {len(examples)} {sentiment} examples")
                data.extend(examples)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load seed examples
    seed_path = os.path.join(RAW_DIR, "seed_examples.json")
    if os.path.exists(seed_path):
        try:
            with open(seed_path, 'r') as f:
                examples = json.load(f)
            print(f"Loaded {len(examples)} seed examples")
            data.extend(examples)
        except Exception as e:
            print(f"Error loading seed examples: {e}")
    
    # Load edge cases
    edge_path = os.path.join(RAW_DIR, "edge_cases.json")
    if os.path.exists(edge_path):
        try:
            with open(edge_path, 'r') as f:
                edge_cases = json.load(f)
            
            # Add edge cases (using primary sentiment)
            for case in edge_cases:
                if "text" in case and "primary_sentiment" in case:
                    data.append({
                        "text": case["text"],
                        "sentiment": case["primary_sentiment"],
                        "is_edge_case": True,
                        "secondary_sentiment": case.get("secondary_sentiment")
                    })
            print(f"Loaded {len(edge_cases)} edge cases")
        except Exception as e:
            print(f"Error loading edge cases: {e}")
    
    # Load conversational examples
    conv_path = os.path.join(RAW_DIR, "conversational_examples.json")
    if os.path.exists(conv_path):
        try:
            with open(conv_path, 'r') as f:
                conversations = json.load(f)
            
            # Extract customer messages with sentiment
            for conv in conversations:
                for turn in conv.get("turns", []):
                    if turn.get("speaker") == "user" and "sentiment" in turn:
                        data.append({
                            "text": turn["text"],
                            "sentiment": turn["sentiment"],
                            "conversation_id": conv.get("conversation_id"),
                            "pattern": conv.get("pattern")
                        })
            print(f"Loaded conversational examples")
        except Exception as e:
            print(f"Error loading conversational examples: {e}")
    
    return data

def normalize_data(examples):
    """Clean and normalize sentiment labels."""
    normalized = []
    
    for ex in examples:
        if not ex.get("text") or not ex.get("sentiment"):
            continue
            
        text = ex["text"].strip()
        sentiment = ex["sentiment"].strip().lower()
        
        # Normalize sentiment labels
        if sentiment in ["urgent negative", "urgent-negative", "urgent_negative"]:
            sentiment = "urgent_negative"
        elif sentiment in ["standard negative", "standard-negative", "standard_negative", "negative"]:
            sentiment = "standard_negative"
        elif sentiment in ["neutral"]:
            sentiment = "neutral"
        elif sentiment in ["positive"]:
            sentiment = "positive"
        else:
            print(f"Unknown sentiment: {sentiment}")
            continue
            
        # Create normalized example
        norm_ex = {
            "text": text,
            "sentiment": sentiment
        }
        
        # Preserve metadata fields
        for field in ["is_edge_case", "secondary_sentiment", "conversation_id", "pattern"]:
            if field in ex:
                norm_ex[field] = ex[field]
                
        normalized.append(norm_ex)
    
    return normalized

def deduplicate_data(examples):
    """Remove duplicate examples based on text."""
    unique_texts = set()
    unique_examples = []
    
    for ex in examples:
        if ex["text"] not in unique_texts:
            unique_texts.add(ex["text"])
            unique_examples.append(ex)
    
    print(f"Removed {len(examples) - len(unique_examples)} duplicates")
    return unique_examples

def balance_data(examples, max_per_category=None):
    """Balance dataset across sentiment categories."""
    # Group by sentiment
    by_sentiment = {}
    for ex in examples:
        sentiment = ex["sentiment"]
        if sentiment not in by_sentiment:
            by_sentiment[sentiment] = []
        by_sentiment[sentiment].append(ex)
    
    # Print original distribution
    print("\nOriginal distribution:")
    for sentiment, examples_list in by_sentiment.items():
        print(f"  {sentiment}: {len(examples_list)}")
    
    # Determine max per category if not specified
    if max_per_category is None:
        max_per_category = min(len(examples) for examples in by_sentiment.values())
    
    # Sample from each category
    balanced = []
    for sentiment, examples_list in by_sentiment.items():
        if len(examples_list) <= max_per_category:
            balanced.extend(examples_list)
        else:
            balanced.extend(random.sample(examples_list, max_per_category))
    
    # Shuffle the balanced dataset
    random.shuffle(balanced)
    
    # Print balanced distribution
    sentiment_counts = Counter(ex["sentiment"] for ex in balanced)
    print("\nBalanced distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count}")
        
    return balanced

def create_splits(examples, test_size=0.15, val_size=0.15):
    """Split data into train, validation, and test sets."""
    # First split off test set
    train_val, test = train_test_split(
        examples,
        test_size=test_size,
        stratify=[ex["sentiment"] for ex in examples],
        random_state=42
    )
    
    # Now split validation from training
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=[ex["sentiment"] for ex in train_val],
        random_state=42
    )
    
    print(f"\nSplit into {len(train)} train, {len(val)} validation, {len(test)} test examples")
    
    # Print distribution in each split
    for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
        counts = Counter(ex["sentiment"] for ex in split)
        print(f"\n{name} distribution:")
        for sentiment, count in counts.items():
            print(f"  {sentiment}: {count}")
    
    return train, val, test

def save_splits(train, val, test):
    """Save the data splits to files."""
    with open(os.path.join(FINAL_DIR, "sentiment_train.json"), 'w') as f:
        json.dump(train, f, indent=2)
        
    with open(os.path.join(FINAL_DIR, "sentiment_validation.json"), 'w') as f:
        json.dump(val, f, indent=2)
        
    with open(os.path.join(FINAL_DIR, "sentiment_test.json"), 'w') as f:
        json.dump(test, f, indent=2)
        
    print(f"\nSaved datasets to {FINAL_DIR}")

def main():
    print("Loading all sentiment data...")
    all_data = load_all_raw_data()
    
    print("\nNormalizing sentiment labels...")
    normalized_data = normalize_data(all_data)
    
    print("\nRemoving duplicates...")
    unique_data = deduplicate_data(normalized_data)
    
    print("\nBalancing dataset across sentiment categories...")
    # Target 50 examples per category (adjust as needed)
    balanced_data = balance_data(unique_data, max_per_category=50)
    
    print("\nSplitting into train/validation/test sets...")
    train, val, test = create_splits(balanced_data)
    
    print("\nSaving final datasets...")
    save_splits(train, val, test)
    
    print("\nComplete! Your datasets now include all sentiment categories.")

if __name__ == "__main__":
    main()