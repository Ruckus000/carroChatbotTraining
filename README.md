# Chatbot Training Framework

This framework provides a comprehensive solution for fine-tuning DistilBERT models to power a multi-flow chatbot system covering towing services, roadside assistance, and service appointment booking.

## Features

- Multi-task hierarchical model training pipeline
- Sophisticated data augmentation for improved robustness
- Specialized fallback and clarification detection
- Advanced entity extraction
- Comprehensive evaluation framework for model testing

## Project Structure

```
chatbot/
├── chatbot_training.py       # Main orchestration script
├── data_augmentation.py      # Data variation and noise functions
├── model_training.py         # DistilBERT fine-tuning functions
├── evaluation.py             # Testing and evaluation functions
├── utils.py                  # Utility functions
└── requirements.txt          # Dependencies
```

## Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Create a JSON file with your conversation data following this structure:

```json
[
  {
    "flow": "towing",
    "intent": "request_tow_basic",
    "input": "I need a tow truck.",
    "response": "I can help with that! Where should they pick up your vehicle and where should it be towed?",
    "context": {"display_map": true},
    "entities": []
  },
  ...
]
```

### 3. Run the Training Pipeline

```bash
python chatbot_training.py --input_data path/to/conversations.json --output_dir ./output --augment_data --train_models
```

### 4. Evaluate Models

```bash
python chatbot_training.py --input_data path/to/conversations.json --output_dir ./output --augment_data --extreme_test
```

## Key Components

### Data Augmentation

The system provides extensive data augmentation techniques:

- Entity variations (missing, reordered, format changes)
- Noise injection (typos, spacing issues, abbreviations)
- Domain-specific text patterns (slang, shorthand)
- Extreme test cases (terse inputs, run-on sentences)

### Hierarchical Model Architecture

The framework trains several specialized models:

1. **Flow Classifier** - Routes requests to the appropriate conversation flow
2. **Intent Classifiers** - Flow-specific models to identify detailed user intentions
3. **Fallback Detector** - Identifies out-of-domain requests
4. **Clarification Detector** - Recognizes ambiguous inputs requiring clarification
5. **Entity Extractor** - Extracts structured information from natural language

### Robustness Testing

Comprehensive evaluation includes:

- Standard classification metrics (precision, recall, F1)
- Specialized robustness metrics for noise tolerance
- Entity extraction quality assessment
- Extreme case handling evaluation

## Command Line Arguments

- `--input_data`: Path to input conversation data JSON file
- `--output_dir`: Directory to save processed data and models
- `--test_size`: Proportion of data to use for testing (default: 0.15)
- `--val_size`: Proportion of data to use for validation (default: 0.15)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--augment_data`: Apply data augmentation techniques
- `--extreme_test`: Generate extreme test cases for robustness testing
- `--train_models`: Train models after preprocessing data

## Example Usage

### Basic Preprocessing Only

```bash
python chatbot_training.py --input_data conversations.json --output_dir ./output
```

### With Data Augmentation

```bash
python chatbot_training.py --input_data conversations.json --output_dir ./output --augment_data
```

### Full Pipeline with Training

```bash
python chatbot_training.py --input_data conversations.json --output_dir ./output --augment_data --train_models
```
