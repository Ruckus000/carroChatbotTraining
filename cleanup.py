#!/usr/bin/env python3
import os
import shutil
import sys

def main():
    """
    Clean up the project by removing obsolete files and updating requirements.txt
    """
    print("Starting cleanup process...")
    
    # Files to delete
    files_to_delete = [
        # Context integration files
        "context-integration.md",
        "context1-implementation.md",
        "context1.js",
        "README-CONTEXT.md",
        "test_context_integration.py",
        "test_context_integration_comprehensive.py",
        "run_context_integration.sh",
        "train_context_models.py",
        "train_context_models copy.py",
        "train_context_models copy.md",
        "evaluate_context_models.py",
        
        # Unused utility scripts
        "chatbot_training.py",
        "model_training.py",
        "evaluation.py",
        "streamlit_app.py",
        "chatbot-booking-process.md",
        "langGraph-Mistral-implementation-plan.md",
        "data_augmentation.py",
        "augment_conversations.py",
        "custom_trainer.py",
        "fix_indent.py",
        "plan2fixshit.md",
        
        # Unused deployment/setup scripts
        "setup.py",
        "deploy.sh",
        "run_chatbot.py",
        "run_training.sh",
    ]
    
    # Directories to delete
    dirs_to_delete = [
        "langgraph_integration"
    ]
    
    # Delete files
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"Skipped (not found): {file}")
    
    # Delete directories
    for directory in dirs_to_delete:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Deleted directory: {directory}")
            except Exception as e:
                print(f"Error deleting directory {directory}: {e}")
        else:
            print(f"Skipped directory (not found): {directory}")
    
    # Update requirements.txt
    requirements = [
        "transformers==4.30.2",
        "torch==2.0.1",
        "datasets==2.12.0",
        "scikit-learn==1.2.2",
        "numpy==1.24.3",
        "seqeval==1.2.2"
    ]
    
    try:
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        print("Updated requirements.txt")
    except Exception as e:
        print(f"Error updating requirements.txt: {e}")
    
    # Create a new README.md
    readme_content = """# Simple NLU System

## Overview
This is a simple Natural Language Understanding (NLU) system that performs intent detection and entity recognition.

## Setup
```bash
pip install -r requirements.txt
```

## Data Format
The training data is stored in `data/nlu_training_data.json` in the following format:
```json
[
  {
    "text": "I need a tow truck",
    "intent": "towing_request_tow",
    "entities": [
      {
        "entity": "service_type",
        "value": "tow truck"
      }
    ]
  },
  ...
]
```

## Training
To train the NLU models:
```bash
python train.py
```

This will generate intent and entity models in the `trained_nlu_model` directory.

## Usage
```python
from inference import NLUInferencer

# Initialize the inferencer
nlu = NLUInferencer()

# Make a prediction
result = nlu.predict("I need a tow truck at 123 Main Street")
print(result)
```

## Testing
To run the integration test:
```bash
python test_integration.py
```
"""
    
    try:
        with open("README.md", "w") as f:
            f.write(readme_content)
        print("Created new README.md")
    except Exception as e:
        print(f"Error creating README.md: {e}")
    
    print("Cleanup complete!")

if __name__ == "__main__":
    main() 