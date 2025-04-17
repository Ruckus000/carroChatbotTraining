#!/usr/bin/env python3
import os
import shutil
import sys


def main():
    """
    Clean up the project by removing obsolete code, data, and documentation,
    leaving only the simplified NLU components.
    """
    print("Starting cleanup process...")

    # Files to explicitly preserve
    files_to_preserve = [
        "train.py",
        "inference.py",
        "test_integration.py",
        "data/nlu_training_data.json",
        "requirements.txt",
        "README.md",
        ".gitignore",
        # Test files
        "test_phase1.py",
        "test_phase2.py",
        "test_phase3.py",
        "test_phase5.py",
    ]

    # Directories to explicitly delete
    dirs_to_delete = [
        "langgraph_integration/",
        "data/context_integration/",
        "output/",
        "tests/",
    ]

    # Delete directories
    for directory in dirs_to_delete:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory, ignore_errors=True)
                print(f"Deleted directory: {directory}")
            except Exception as e:
                print(f"Error deleting directory {directory}: {e}")
        else:
            print(f"Skipped directory (not found): {directory}")

    # Find all files in the project
    all_files = []
    for root, dirs, files in os.walk("."):
        # Skip .git directory and trained_nlu_model
        if ".git" in root or "trained_nlu_model" in root or "__pycache__" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file).replace("./", "")
            all_files.append(file_path)

    # Delete files that are not in the preserve list
    for file in all_files:
        if (
            file not in files_to_preserve
            and "cleanup.py" not in file
            and file != "plan3fixshit.md"
        ):
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")

    # Update requirements.txt
    requirements = [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
        "seqeval>=1.2.0",
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

## Inference
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
