#!/usr/bin/env python3
"""
Script to merge generated training examples with the main training data.
"""

import json
import argparse
import os

def merge_training_data(main_file, generated_file, output_file=None):
    """
    Merge training data from generated examples into the main training data.
    
    Args:
        main_file: Path to the main training data file
        generated_file: Path to the generated examples file
        output_file: Path to save the merged data (defaults to overwriting main_file)
    """
    # Default output to main file if not specified
    if output_file is None:
        output_file = main_file
    
    # Load main training data
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            main_data = json.load(f)
        print(f"Loaded {len(main_data)} examples from {main_file}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading main training file: {e}")
        return
    
    # Load generated examples
    try:
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        print(f"Loaded {len(generated_data)} examples from {generated_file}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading generated examples file: {e}")
        return
    
    # Merge data
    merged_data = main_data + generated_data
    print(f"Combined data has {len(merged_data)} examples")
    
    # Save merged data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"Saved merged data to {output_file}")
    except IOError as e:
        print(f"Error saving merged data: {e}")
    
    # Print intent distribution
    intent_counts = {}
    for example in merged_data:
        intent = example.get('intent', 'unknown')
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nIntent distribution in merged data:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent}: {count} examples")

def main():
    parser = argparse.ArgumentParser(description="Merge generated examples with main training data")
    parser.add_argument("--main", type=str, default="data/nlu_training_data.json",
                       help="Path to main training data file")
    parser.add_argument("--generated", type=str, default="data/generated_examples.json",
                       help="Path to generated examples file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for merged data (defaults to overwriting main file)")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup of main file before overwriting")
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup and args.output is None:
        backup_file = f"{args.main}.backup"
        try:
            with open(args.main, 'r', encoding='utf-8') as src:
                with open(backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"Created backup at {backup_file}")
        except IOError as e:
            print(f"Error creating backup: {e}")
            return
    
    merge_training_data(args.main, args.generated, args.output)

if __name__ == "__main__":
    main() 