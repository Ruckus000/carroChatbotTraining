# Simple NLU System for Chatbot

## Overview

This repository contains a **simplified** Natural Language Understanding (NLU) system built using Python and Hugging Face Transformers (specifically DistilBERT). It provides:

1.  **Intent Detection:** What is the user trying to _do_? (e.g., request a tow, ask about battery)
2.  **Entities:** What are the key pieces of information in the text? (e.g., location, vehicle type)
3.  **Sentiment Analysis:** What is the emotional tone of the message? (positive, negative, neutral)
4.  **Dialog Management:** A unified dialog management system that handles conversation state and determines appropriate actions
5.  **Response Generation:** Generates appropriate responses based on the dialog state and determined actions

The system features a clean architecture with dependency injection, making it flexible and testable. It also leverages Apple Silicon GPU acceleration (MPS) when available for both training and inference.

---

## Core Functionality

- **Intent Detection:** Classifies the input text into a predefined set of intents. Intents are expected to follow a `flow_subintent` naming convention (e.g., `towing_request_tow_basic`, `roadside_request_battery`). If the model's confidence is below a threshold (currently hardcoded at 0.5 in `inference.py`), it defaults to a `fallback_low_confidence` intent.
- **Entity Recognition:** Identifies and extracts named entities from the text. It uses the **BIO (Beginning, Inside, Outside)** tagging scheme. For example, "123 Main Street" might be tagged as `B-pickup_location I-pickup_location I-pickup_location`. The inference script then groups these tags into entity dictionaries like `{"entity": "pickup_location", "value": "123 Main Street"}`.
- **Sentiment Analysis:** Analyzes the emotional tone of the message and provides a sentiment label (positive, negative, neutral) with a confidence score. This information is used to adapt responses and optimize dialog flow, especially for urgent situations.
- **Dialog Management:** The `DialogManager` class manages conversation state for multiple conversations, determines appropriate actions based on intents, entities, and sentiment, and coordinates the flow of the conversation.
- **Response Generation:** The `ResponseGenerator` class generates human-readable responses based on the determined action, current dialog state, and sentiment context.

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI configuration
├── data/
│   └── nlu_training_data.json  # THE ONLY TRAINING DATA FILE
├── trained_nlu_model/          # Output directory for trained models
│   ├── intent_model/           # Intent classification model files
│   └── entity_model/           # Entity recognition model files
├── .gitignore                  # Standard Git ignore file
├── inference.py                # Loads models and performs NLU prediction
├── dialog_manager.py           # Manages conversation state and determines actions
├── response_generator.py       # Generates human-readable responses
├── api.py                      # FastAPI server for exposing functionality
├── README.md                   # This file
├── API_README.md               # API documentation for frontend integration
├── requirements.txt            # Full Python dependencies including dev/testing
├── requirements-api.txt        # Runtime dependencies for the API server
├── test_integration.py         # Basic integration test for NLUInferencer
├── test_api_integration.py     # API integration tests
├── test_dialog_manager_unified.py # Tests for the unified DialogManager
└── train.py                    # Script to train the NLU models
```

---

## Architecture

The system follows a modular architecture with clear separation of concerns:

1. **NLU Layer (`inference.py`):** Handles text processing, intent detection, entity extraction, and sentiment analysis.
2. **Dialog Management Layer (`dialog_manager.py`):** Manages conversation state, determines actions based on intents/entities/sentiment.
3. **Response Generation Layer (`response_generator.py`):** Creates natural language responses based on actions and sentiment context.
4. **API Layer (`api.py`):** Exposes the functionality via a REST API using FastAPI.

Key architectural features:

- **Dependency Injection:** The DialogManager requires an NLUInferencer instance to be provided at initialization, allowing for loose coupling and easier testing.
- **State Management:** Conversation state is maintained per conversation ID, allowing for multiple simultaneous conversations.
- **Action Determination:** Based on the current state, NLU results, and sentiment analysis, the DialogManager determines appropriate actions.
- **Templated Responses:** The ResponseGenerator uses templates to create varied and context-appropriate responses, with special variations for negative sentiment.
- **Hardware Acceleration:** The system automatically detects and uses Apple Silicon GPU (MPS) for inference and training when available.

---

## Apple Silicon GPU Acceleration

The system automatically detects and uses Apple Silicon GPU (MPS) for both training and inference:

- **Inference:** The NLUInferencer automatically detects MPS availability and moves models to the MPS device.
- **Training:** The training script detects MPS and configures the training process to use it, resulting in significantly faster training times.

No special configuration is needed - if you're running on a Mac with Apple Silicon (M1/M2/M3/M4), the system will automatically use hardware acceleration.

---

## Setup

1.  **Clone the repo:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (Seriously, do it):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
    **IMPORTANT:** Always activate the virtual environment when working on this project. This isolates your dependencies and avoids conflicts with system Python packages.
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This installs all required dependencies for development, training, and testing.

4.  **Code Formatting:**
    ```bash
    black .
    ```
    Run this to format your code according to the project style guidelines before committing changes.

---

## Training Data Format (`data/nlu_training_data.json`)

This is the **only** file used for training. It's a JSON list of dictionaries, each representing one training example:

```json
[
  {
    "text": "My 2018 Honda Civic needs a tow from 123 Main Street",
    "intent": "towing_request_tow_vehicle", // Format: flow_subintent
    "entities": [
      {
        "entity": "vehicle_year", // The type of entity
        "value": "2018" // The exact text corresponding to the entity
      },
      {
        "entity": "vehicle_make",
        "value": "Honda"
      },
      {
        "entity": "vehicle_model",
        "value": "Civic"
      },
      {
        "entity": "pickup_location",
        "value": "123 Main Street" // MUST match the text exactly
      }
    ]
  },
  {
    "text": "What is the weather?",
    "intent": "fallback_out_of_scope_weather",
    "entities": [] // Empty list if no entities
  }
  // ... more examples
]
```

---

## Training the Models

To train both the intent and entity recognition models:

```bash
python train.py
```

**What it Does:**

1.  Loads data from `data/nlu_training_data.json`.
2.  Splits data into training and validation sets.
3.  **Trains Intent Model:** Fine-tunes a `DistilBertForSequenceClassification` model on the `text` and `intent` fields.
4.  **Trains Entity Model:** Fine-tunes a `DistilBertForTokenClassification` model using the `text` and `entities` fields (converting entities to BIO tags internally).
5.  Saves the trained models, tokenizers, and necessary configuration/mapping files into `./trained_nlu_model/intent_model/` and `./trained_nlu_model/entity_model/`.
6.  **Hardware Acceleration:** Automatically uses Apple Silicon GPU (MPS) if available, significantly improving training speed.

---

## Running the API Server

To run the API server:

```bash
# Use port 8001 to avoid conflicts with the Docker container that uses port 8000
# You can also set a custom port using the PORT environment variable
export PORT=8001 && python api.py
```

This starts a FastAPI server on http://localhost:8001 that provides:

- `/api/health` - Health check endpoint
- `/api/nlu` - NLU processing endpoint (includes sentiment analysis)
- `/api/dialog` - Dialog processing endpoint that maintains conversation state

**IMPORTANT NOTE**: There is a Docker container running on port 8000 with an older version of the API. For development and testing, use the local API on port 8001 as shown above. This ensures you're working with the latest version of the code.

See the `API_README.md` file for detailed API documentation and integration examples.

---

## Sentiment Analysis

The system now includes sentiment analysis as part of the NLU pipeline:

- **Technical Implementation:** Uses Hugging Face's sentiment analysis pipeline with a pre-trained DistilBERT model.
- **Integration:** Sentiment results (label and score) are included in the NLU output and stored in the dialog state.
- **Dialog Flow Impact:** High negative sentiment combined with certain intents (like towing requests) can trigger expedited "urgent" flows.
- **Response Adaptation:** The response generator uses sentiment information to provide more empathetic, urgent-focused responses for negative sentiment situations.

This feature is particularly valuable for automotive assistance scenarios where user urgency and emotional state can significantly impact the required service level.

---

## Testing

Several test suites are available to verify different components of the system:

```bash
# Basic NLU integration test
python test_integration.py

# Test the unified DialogManager
python -m unittest test_dialog_manager_unified.py

# Run API integration tests (requires the API server to be running)
python -m pytest test_api_integration.py
```

---

## CI Workflow

The GitHub Actions workflow in `.github/workflows/ci.yml` automatically runs linters (`flake8`, `black`, `isort`) and the test scripts on pushes and pull requests to the `main` and `development` branches.

---

## Technology Stack

- **Python:** Core language
- **Hugging Face Transformers:** For DistilBERT models and tokenizers
- **PyTorch:** Backend for Transformers
- **FastAPI:** REST API framework
- **Uvicorn:** ASGI server for FastAPI
- **Scikit-learn:** For data splitting
- **Seqeval:** For entity recognition metrics (used during training)
- **NumPy:** Numerical operations
- **Black:** Code formatting
- **Pytest:** For testing
