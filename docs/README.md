# Simple NLU System for Chatbot

## Overview

This repository contains a **simplified** Natural Language Understanding (NLU) system built using Python and Hugging Face Transformers (specifically DistilBERT). It provides:

1.  **Intent Detection:** What is the user trying to _do_? (e.g., request a tow, ask about battery)
2.  **Entities:** What are the key pieces of information in the text? (e.g., location, vehicle type)
3.  **Dialog Management:** A unified dialog management system that handles conversation state and determines appropriate actions
4.  **Response Generation:** Generates appropriate responses based on the dialog state and determined actions

The system features a clean architecture with dependency injection, making it flexible and testable.

---

## Core Functionality

- **Intent Detection:** Classifies the input text into a predefined set of intents. Intents are expected to follow a `flow_subintent` naming convention (e.g., `towing_request_tow_basic`, `roadside_request_battery`). If the model's confidence is below a threshold (currently hardcoded at 0.5 in `inference.py`), it defaults to a `fallback_low_confidence` intent.
- **Entity Recognition:** Identifies and extracts named entities from the text. It uses the **BIO (Beginning, Inside, Outside)** tagging scheme. For example, "123 Main Street" might be tagged as `B-pickup_location I-pickup_location I-pickup_location`. The inference script then groups these tags into entity dictionaries like `{"entity": "pickup_location", "value": "123 Main Street"}`.
- **Dialog Management:** The `DialogManager` class manages conversation state for multiple conversations, determines appropriate actions based on intents and entities, and coordinates the flow of the conversation.
- **Response Generation:** The `ResponseGenerator` class generates human-readable responses based on the determined action and current dialog state.

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

1. **NLU Layer (`inference.py`):** Handles text processing, intent detection, and entity extraction.
2. **Dialog Management Layer (`dialog_manager.py`):** Manages conversation state, determines actions based on intents/entities.
3. **Response Generation Layer (`response_generator.py`):** Creates natural language responses based on actions.
4. **API Layer (`api.py`):** Exposes the functionality via a REST API using FastAPI.

Key architectural features:

- **Dependency Injection:** The DialogManager requires an NLUInferencer instance to be provided at initialization, allowing for loose coupling and easier testing.
- **State Management:** Conversation state is maintained per conversation ID, allowing for multiple simultaneous conversations.
- **Action Determination:** Based on the current state and NLU results, the DialogManager determines appropriate actions.
- **Templated Responses:** The ResponseGenerator uses templates to create varied and context-appropriate responses.

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

---

## Running the API Server

To run the API server:

```bash
# Use port 8001 to avoid conflicts with the Docker container that uses port 8000
python -c "import uvicorn; import api; uvicorn.run(api.app, host='127.0.0.1', port=8001)"
```

This starts a FastAPI server on http://localhost:8001 that provides:

- `/api/health` - Health check endpoint
- `/api/nlu` - NLU processing endpoint
- `/api/dialog` - Dialog processing endpoint that maintains conversation state

**IMPORTANT NOTE**: There is a Docker container running on port 8000 with an older version of the API. For development and testing, use the local API on port 8001 as shown above. This ensures you're working with the latest version of the code.

See the `API_README.md` file for detailed API documentation and integration examples.

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
