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

## Streamlit Demo Application

The project includes a Streamlit web application that allows you to interact with the trained models and compare different processing modes:

```bash
streamlit run streamlit_app.py
```

### Features of the Demo:

1. **Standard (Original) Mode**: Basic flow and intent classification pipeline
2. **Context-Aware (Enhanced) Mode**: Advanced assistant that can:
   - Track conversation context across multiple turns
   - Detect topic/flow changes
   - Identify contradictions in user statements
   - Recognize negations and cancellations
   - Maintain entity memory throughout the conversation

### Debug Information

The application includes a debug panel that shows:

- Flow and intent predictions with confidence scores
- Detected entities
- Context switches, contradictions, and negations
- Full conversation context history

## LangGraph Integration

Recent updates include experimental integration with LangGraph for even more sophisticated conversation management:

```bash
python run_chatbot.py
```

This integration enables:

- Multi-agent collaboration between specialized components
- More nuanced state management
- Enhanced reasoning capabilities
