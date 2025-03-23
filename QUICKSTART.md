# DistilBERT Chatbot Training Framework Quickstart

## Step 1: Setup Environment

First, set up the Python environment and make scripts executable:

```bash
cd /path/to/carroChatbotTraining/chatbot
chmod +x setup.sh
./setup.sh
```

This will create a virtual environment, install dependencies, and make the training script executable.

## Step 2: Understand Sample Data

The sample data in `data/sample_conversations.json` contains examples for:
- Towing service requests
- Roadside assistance
- Service appointment booking
- Fallback handling
- Clarification requests

Review the `data/README.md` file for details on the data structure.

## Step 3: Run Training Pipeline

You can run the training in different modes:

### Basic Preprocessing

```bash
python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output
```

### With Data Augmentation

```bash
python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output --augment_data
```

### Full Pipeline with Training and Evaluation

```bash
python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output --augment_data --train_models --evaluate_models
```

### Run Everything with the Convenience Script

```bash
./run_training.sh
```

## Step 4: Check Results

After training, check the `output` directory for:

- `datasets/`: Prepared training data for each model
- `models/`: Trained model files and configurations
- `evaluation/`: Performance metrics and visualizations

## Step 5: Customize for Production

For a production-quality chatbot:
1. Add more training examples (100+ per intent)
2. Uncomment actual model training code in `model_training.py`
3. Add domain-specific entities and intents
4. Include more extreme test cases
5. Fine-tune model hyperparameters

## Model Architecture

The framework trains these models:
1. **Flow Classifier**: Routes to towing, roadside, or appointment flows
2. **Intent Classifiers**: Determines specific intents within each flow
3. **Entity Extractor**: Extracts structured information (locations, dates, etc.)
4. **Fallback Detector**: Identifies out-of-domain queries
5. **Clarification Detector**: Recognizes ambiguous requests

## Command Line Arguments

The main script supports these arguments:
- `--input_data`: Path to conversation data JSON
- `--output_dir`: Directory for results
- `--augment_data`: Apply data augmentation
- `--train_models`: Train the models after preprocessing
- `--evaluate_models`: Run evaluation after training
- `--extreme_test`: Generate challenging test cases
- `--test_size`: Proportion for test set (default: 0.15) 
- `--val_size`: Proportion for validation set (default: 0.15)
- `--random_seed`: Seed for reproducibility (default: 42)