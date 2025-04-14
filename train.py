import json
import os
import logging
import torch
import numpy as np
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seqeval.metrics

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "data/nlu_training_data.json"
OUTPUT_DIR = "./trained_nlu_model"
INTENT_MODEL_DIR = os.path.join(OUTPUT_DIR, "intent_model")
ENTITY_MODEL_DIR = os.path.join(OUTPUT_DIR, "entity_model")
INTENT_CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "intent_model_checkpoints")
ENTITY_CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "entity_model_checkpoints")

# Ensure directories exist
os.makedirs(INTENT_MODEL_DIR, exist_ok=True)
os.makedirs(ENTITY_MODEL_DIR, exist_ok=True)
os.makedirs(INTENT_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(ENTITY_CHECKPOINTS_DIR, exist_ok=True)

def load_data(filepath):
    """Load training data from JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} examples from {filepath}")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file {filepath}: {e}")
                raise
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_intent_model(training_data, validation_data):
    """Train the intent classification model."""
    # TODO: Implement intent model training as per Phase 2 requirements
    logger.info("Intent model training placeholder - would train on actual data")
    
    # Save intent model files
    placeholder_intent2id = {"placeholder": 0}
    with open(os.path.join(INTENT_MODEL_DIR, "intent2id.json"), "w") as f:
        json.dump(placeholder_intent2id, f, indent=2)
    
    logger.info(f"Intent model trained and saved to {INTENT_MODEL_DIR}")
    return True

def train_entity_model(training_data, validation_data):
    """Train the entity extraction model."""
    # TODO: Implement entity model training as per Phase 2 requirements
    logger.info("Entity model training placeholder - would train on actual data")
    
    # Save entity model files
    placeholder_tag2id = {"O": 0, "B-placeholder": 1, "I-placeholder": 2}
    with open(os.path.join(ENTITY_MODEL_DIR, "tag2id.json"), "w") as f:
        json.dump(placeholder_tag2id, f, indent=2)
    
    logger.info(f"Entity model trained and saved to {ENTITY_MODEL_DIR}")
    return True

def main():
    logger.info("Starting model training")
    
    try:
        # Load data
        all_data = load_data(DATA_PATH)
        
        # Split data
        train_data, val_data = train_test_split(
            all_data, test_size=0.2, random_state=42
        )
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")
        
        # Train intent model
        intent_success = train_intent_model(train_data, val_data)
        
        # Train entity model
        entity_success = train_entity_model(train_data, val_data)
        
        if intent_success and entity_success:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed")
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 