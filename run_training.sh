#!/bin/bash

# Create output directory
mkdir -p output

# Run preprocessing only
echo "Running data preprocessing..."
python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output

# Run with data augmentation
echo "Running with data augmentation..."
python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output --augment_data

# Run full pipeline with training and evaluation
echo "Running full training pipeline..."
python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output --augment_data --train_models --evaluate_models

echo "Chatbot training complete!"
