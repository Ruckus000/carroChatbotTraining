# Simple NLU System

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
