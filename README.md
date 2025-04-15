# Simple NLU System

## Overview

This is a simple Natural Language Understanding (NLU) system that performs intent detection and entity recognition for vehicle service requests. The system uses two separate transformer-based models:

1. **Intent Classification Model**: Detects the user's intent (towing, roadside assistance, appointment, etc.)
2. **Entity Recognition Model**: Extracts relevant entities like locations, vehicle information, and service types

## Setup

```bash
pip install -r requirements.txt
```

## Training Data

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
  }
]
```

### Intent Categories

The current training data includes several intent categories:

- **Towing**: `towing_request_tow`, `towing_flatbed_request`, etc.
- **Roadside Assistance**: `roadside_request_battery`, `roadside_request_roadside_fuel`, etc.
- **Appointments**: `appointment_book_service`, `appointment_seasonal_recommendation`, etc.
- **Fallback**: `fallback_out_of_domain`, `fallback_low_confidence`, etc.

### Entity Types

Common entity types in the dataset:

- `pickup_location`: Location for towing or service
- `vehicle_make`: Make of the vehicle (Honda, Toyota, etc.)
- `vehicle_model`: Model of the vehicle (Civic, Corolla, etc.)
- `service_type`: Type of service requested
- `time`: Time information
- `date`: Date information

## Training

To train the NLU models:

```bash
python train.py
```

This will:

1. Load and standardize the training data
2. Split into training and validation sets
3. Train separate models for intent classification and entity recognition
4. Save the models to the `trained_nlu_model` directory

### Training Parameters

- Training runs for 2 epochs by default
- Models run on CPU only (`no_cuda=True`)
- Uses DistilBERT as the base model for lightweight performance

## Usage

```python
from inference import NLUInferencer

# Initialize the inferencer
nlu = NLUInferencer()

# Make a prediction
result = nlu.predict("I need a tow truck at 123 Main Street")
print(result)

# Example output:
# {
#   "text": "I need a tow truck at 123 Main Street",
#   "intent": {
#     "name": "towing_request_tow",
#     "confidence": 0.87
#   },
#   "entities": [
#     {
#       "entity": "service_type",
#       "value": "tow truck"
#     },
#     {
#       "entity": "pickup_location",
#       "value": "123 Main Street"
#     }
#   ]
# }
```

## Testing

To run the integration test:

```bash
python test_integration.py
```

## Structure

This project has a clean, minimalist structure:

- `data/nlu_training_data.json`: Training data
- `train.py`: Script to train both intent and entity models
- `inference.py`: Contains the NLUInferencer class for making predictions
- `test_integration.py`: Tests the inference on sample inputs
- `trained_nlu_model/`: Directory containing trained models
  - `intent_model/`: The intent classification model
  - `entity_model/`: The entity recognition model

## About the Implementation

This implementation focuses on simplicity and efficiency:

- Uses the BIO tagging scheme for entity recognition
- Provides fallback intents for low confidence predictions
- Handles entity grouping from BIO tags automatically
- Runs efficiently on CPU-only environments

## Expanding the Training Data

### Suggestions for Improvement

1. **Add More Diverse Examples**:

   - Include multiple ways to express the same intent
   - Add examples with different entity values
   - Include examples with multiple entities

2. **Balance Intent Categories**:

   - Ensure each intent has a sufficient number of examples
   - Add more examples for less common intents

3. **Expand Entity Coverage**:

   - Add more vehicle makes and models
   - Include different location formats (addresses, landmarks, GPS coordinates)
   - Add time expressions (morning, afternoon, specific times)

4. **Add Edge Cases**:

   - Misspellings and typos
   - Incomplete or ambiguous requests
   - Requests with irrelevant information

5. **Data Generation Techniques**:
   - Template-based generation with slot filling
   - Paraphrasing existing examples
   - Back-translation (translate to another language and back)
   - Using LLMs to generate variations of existing examples

### Example Data Expansion

To add new examples, append to the `data/nlu_training_data.json` file:

```json
{
  "text": "My Honda Civic broke down at 456 Oak Street, I need a tow truck ASAP",
  "intent": "towing_request_tow",
  "entities": [
    { "entity": "vehicle_make", "value": "Honda" },
    { "entity": "vehicle_model", "value": "Civic" },
    { "entity": "pickup_location", "value": "456 Oak Street" }
  ]
}
```

After adding new data, retrain the models using `python train.py`.
