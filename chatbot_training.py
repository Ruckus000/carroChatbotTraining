#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for fine-tuning DistilBERT models for a multi-flow chatbot.
This script orchestrates the data processing, augmentation, and model training.
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Local imports
from data_augmentation import augment_conversation_data
from model_training import train_all_models
from utils import prepare_for_distilbert, save_dataset, load_json_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DistilBERT models for chatbot')
    
    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to input conversation data JSON file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save processed data and models')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Proportion of data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--augment_data', action='store_true',
                        help='Apply data augmentation techniques')
    parser.add_argument('--extreme_test', action='store_true',
                        help='Generate extreme test cases for robustness testing')
    parser.add_argument('--train_models', action='store_true',
                        help='Train models after preprocessing data')
    parser.add_argument('--evaluate_models', action='store_true',
                        help='Run evaluation after training models')
    parser.add_argument('--no_mps', action='store_true',
                        help='Disable MPS (Metal Performance Shaders) for training')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Force CPU usage if requested
    if args.no_mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Force torch to use CPU before importing
        import torch
        torch.device('cpu')
        # Set the default device to CPU
        torch._C._set_default_tensor_type(torch.FloatTensor)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading conversation data from {args.input_data}...")
    conversations = load_json_data(args.input_data)
    
    # Check if we have fallback examples to incorporate
    fallback_path = 'data/fallback_flow_examples.json'
    if os.path.exists(fallback_path):
        logger.info(f"Loading fallback flow examples from {fallback_path}")
        try:
            with open(fallback_path, 'r') as f:
                fallback_examples = json.load(f)
            logger.info(f"Loaded {len(fallback_examples)} fallback flow examples")
        except Exception as e:
            logger.error(f"Error loading fallback examples: {e}")
            fallback_examples = []
    else:
        fallback_examples = []
    
    if args.augment_data:
        logger.info("Applying data augmentation techniques...")
        augmented_conversations = augment_conversation_data(
            conversations, 
            extreme_test=args.extreme_test
        )
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(augmented_conversations)
        
        # Remove duplicate inputs
        df = df.drop_duplicates(subset=['input'])
    else:
        # Use original data without augmentation
        df = pd.DataFrame(conversations)
    
    # Log dataset statistics
    logger.info(f"Total examples: {len(df)}")
    logger.info(f"Flow distribution:\n{df['flow'].value_counts()}")
    logger.info(f"Intent distribution (top 10):\n{df['intent'].value_counts().head(10)}") # Show top 10 intents
    logger.info(f"Examples with entities: {len(df[df['entities'].apply(lambda x: len(x) > 0)])}")
    
    # Calculate complexity metrics
    df['word_count'] = df['input'].apply(lambda x: len(str(x).split()))
    df['entity_count'] = df['entities'].apply(len)
    
    logger.info(f"Input complexity metrics:")
    logger.info(f"Average words per input: {df['word_count'].mean():.2f}")
    logger.info(f"Average entities per input: {df['entity_count'].mean():.2f}")
    logger.info(f"Max words in input: {df['word_count'].max()}")
    logger.info(f"Max entities in input: {df['entity_count'].max()}")
    
    # Check if we can stratify by flow
    flow_counts = df['flow'].value_counts()
    can_stratify = all(count >= 2 for count in flow_counts)
    
    # Split data into training, validation, and test sets
    stratify_param = df[['flow']] if can_stratify else None
    if not can_stratify:
        logger.warning("Cannot stratify by flow as some classes have too few examples. Using random split instead.")
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=args.val_size + args.test_size,
        random_state=args.random_seed, 
        stratify=stratify_param  # Stratify by flow to ensure representation
    )
    
    val_size_relative = args.val_size / (args.val_size + args.test_size)
    
    # For the second split, check if we can stratify temp_df
    temp_flow_counts = temp_df['flow'].value_counts()
    can_stratify_temp = all(count >= 2 for count in temp_flow_counts)
    temp_stratify_param = temp_df[['flow']] if can_stratify_temp else None
    
    if not can_stratify_temp and can_stratify:
        logger.warning("Cannot stratify second split by flow. Using random split for validation/test division.")
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=1-val_size_relative,
        random_state=args.random_seed,
        stratify=temp_stratify_param
    )
    
    logger.info(f"Split data into: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
    
    # Create extreme test set if requested
    if args.extreme_test:
        extreme_test_df = df[df['input'].apply(lambda x: 
            any(marker in str(x).lower() for marker in 
                ['asap', 'pls', 'hlp', '@', '&', 'batt', 'tow now', 'no go', 'weird', 'broke down'])
        )]
        logger.info(f"Created extreme test set with {len(extreme_test_df)} examples")
    else:
        extreme_test_df = pd.DataFrame()
    
    # Create datasets for different training tasks
    data_types = ["intent", "flow", "entity", "fallback", "clarification"]
    
    # Create output directory for datasets
    dataset_dir = os.path.join(args.output_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    
    for data_type in data_types:
        logger.info(f"Preparing {data_type} classification data...")
        
        train_data = prepare_for_distilbert(train_df, data_type)
        val_data = prepare_for_distilbert(val_df, data_type)
        test_data = prepare_for_distilbert(test_df, data_type)
        
        save_dataset(train_data, os.path.join(dataset_dir, f"{data_type}_classification_train.json"))
        save_dataset(val_data, os.path.join(dataset_dir, f"{data_type}_classification_val.json"))
        save_dataset(test_data, os.path.join(dataset_dir, f"{data_type}_classification_test.json"))
        
        if not extreme_test_df.empty:
            extreme_test_data = prepare_for_distilbert(extreme_test_df, data_type)
            save_dataset(extreme_test_data, os.path.join(dataset_dir, f"{data_type}_classification_extreme_test.json"))
    
    logger.info(f"All datasets prepared and saved to {dataset_dir}")
    
    # Directly add fallback examples to the intent classification dataset
    if fallback_examples:
        logger.info(f"Adding {len(fallback_examples)} fallback examples to intent classification dataset")
        intent_train_file = os.path.join(dataset_dir, 'intent_classification_train.json')
        
        # Load existing intent dataset
        with open(intent_train_file, 'r') as f:
            intent_train_data = json.load(f)
        
        # Add fallback examples (they're already in the correct format)
        intent_train_data.extend(fallback_examples)
        
        # Save the updated dataset
        with open(intent_train_file, 'w') as f:
            json.dump(intent_train_data, f, indent=2)
        
        logger.info(f"Updated intent classification dataset with fallback examples")
    
    # Train models if requested
    if args.train_models:
        logger.info("Starting model training...")
        model_dir = os.path.join(args.output_dir, "models")
        train_all_models(dataset_dir, model_dir)
    
    # Run evaluation after training if requested
    if args.evaluate_models:
        from evaluation import evaluate_model_pipeline
        logger.info("Running post-training evaluation...")
        eval_output_dir = os.path.join(args.output_dir, "evaluation")
        evaluate_model_pipeline(
            dataset_dir, 
            model_dir, 
            eval_output_dir
        )
        logger.info(f"Evaluation results saved to {eval_output_dir}")
        logger.info("Model training completed!")
    
    logger.info("All processing completed successfully!")

if __name__ == "__main__":
    main()
