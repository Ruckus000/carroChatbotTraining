# Simplified NLU System

A streamlined Natural Language Understanding (NLU) system that performs both intent classification and entity extraction for conversational AI applications. Built with PyTorch and Transformers, this system uses a simplified architecture with two separate models - one for intent detection and one for entity recognition.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/simplified-nlu.git
cd simplified-nlu
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The system uses a simple JSON format for training data, stored in `data/nlu_training_data.json`. Each example has the following structure:

```json
{
  "text": "I need a tow truck at 123 Main Street for my Honda Civic",
  "intent": "towing_request_tow",
  "entities": [
    {
      "entity": "pickup_location",
      "value": "123 Main Street"
    },
    {
      "entity": "vehicle_make",
      "value": "Honda"
    },
    {
      "entity": "vehicle_model",
      "value": "Civic"
    }
  ]
}
```

### Format Details:

- `text`: The input text to analyze (string)
- `intent`: The intent label for the entire utterance (string)
- `entities`: List of entity objects, each with:
  - `entity`: The entity type/label
  - `value`: The extracted entity value

## Training

To train both intent and entity models:

```bash
python train.py
```

This script will:

1. Load training data from `data/nlu_training_data.json`
2. Split data into training and validation sets
3. Train separate models for intent classification and entity recognition
4. Save trained models to `./trained_nlu_model/intent_model` and `./trained_nlu_model/entity_model`

You can verify the models were trained correctly by running:

```bash
python test_phase2.py
```

## Inference

To use the trained models for inference:

```python
from inference import NLUInferencer

# Initialize the inferencer
inferencer = NLUInferencer()

# Make predictions
result = inferencer.predict("I need a tow truck at 123 Main Street for my Honda Civic")
print(f"Intent: {result['intent']['name']} ({result['intent']['confidence']:.4f})")
print(f"Entities: {result['entities']}")
```

### Output Format:

The `predict` method returns a dictionary with:

- `text`: The original input text
- `intent`: A dictionary containing:
  - `name`: The predicted intent label
  - `confidence`: Confidence score (0-1)
- `entities`: A list of extracted entities, each with:
  - `entity`: The entity type
  - `value`: The extracted value

## Testing

The system includes several test scripts:

```bash
# Test data preparation
python test_phase1.py

# Test model training
python test_phase2.py

# Test inference implementation
python test_phase3.py

# Test integration of all components
python test_integration.py
```

To run all tests:

```bash
python -m unittest discover
```

## Project Structure

```
.
├── data/
│   └── nlu_training_data.json     # Consolidated training data
├── trained_nlu_model/
│   ├── intent_model/              # Intent classification model
│   └── entity_model/              # Entity recognition model
├── train.py                       # Training script
├── inference.py                   # Inference implementation
├── test_*.py                      # Test files
├── requirements.txt               # Dependencies
└── README.md                      # This file
```
