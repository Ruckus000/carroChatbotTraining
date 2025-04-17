# Simple NLU System for Chatbot

## Overview

This repository contains a **simplified** Natural Language Understanding (NLU) system built using Python and Hugging Face Transformers (specifically DistilBERT). Its sole purpose is to take a piece of text (like user input to a chatbot) and figure out:

1.  **Intent:** What is the user trying to _do_? (e.g., request a tow, ask about battery)
2.  **Entities:** What are the key pieces of information in the text? (e.g., location, vehicle type)

It does **NOT** handle dialog state management, response generation, complex context tracking, or anything beyond this core NLU task. The training process generates two primary models: one for intent classification and one for entity recognition (using token classification with BIO tagging).

---

## Core Functionality

- **Intent Detection:** Classifies the input text into a predefined set of intents. Intents are expected to follow a `flow_subintent` naming convention (e.g., `towing_request_tow_basic`, `roadside_request_battery`). If the model's confidence is below a threshold (currently hardcoded at 0.5 in `inference.py`), it defaults to a `fallback_low_confidence` intent.
- **Entity Recognition:** Identifies and extracts named entities from the text. It uses the **BIO (Beginning, Inside, Outside)** tagging scheme. For example, "123 Main Street" might be tagged as `B-pickup_location I-pickup_location I-pickup_location`. The inference script then groups these tags into entity dictionaries like `{"entity": "pickup_location", "value": "123 Main Street"}`.

---

## Project Structure (What Actually Matters)

```
.
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI configuration
├── data/
│   └── nlu_training_data.json # THE ONLY TRAINING DATA FILE
├── trained_nlu_model/        # Output directory for trained models
│   ├── intent_model/         # Intent classification model files
│   └── entity_model/         # Entity recognition (token classification) model files
├── .gitignore                # Standard Git ignore file
├── inference.py              # Loads models and performs NLU prediction
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── test_integration.py       # Basic integration test for NLUInferencer
├── test_phase*.py            # Historical/developmental test scripts (ignore for usage)
└── train.py                  # Script to train the NLU models
```

**Gone:** `cleanup.py`, `dialog_manager.py`, `response_generator.py`, `plan*.md`, `imp-rules.md`, and other older test/utility/data files are remnants or process artifacts and **not part of the core NLU system described here.**

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

    This installs `transformers`, `torch`, `datasets`, `scikit-learn`, `numpy`, `seqeval`, and `black` for code formatting. **CPU is assumed.** Training/inference are configured for CPU execution.

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

**CRITICAL:**

- `text`: The raw input string.
- `intent`: A single string identifying the user's goal, using `_` to separate flow/sub-intent.
- `entities`: A list of dictionaries. Each dictionary MUST have `entity` (type name) and `value` (the exact text span from `text`).

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

**Output:** The `./trained_nlu_model/` directory will contain:

- `intent_model/`: Files for the intent classifier (`pytorch_model.bin` or `model.safetensors`, `config.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`, `intent2id.json`).
- `entity_model/`: Files for the entity recognizer (similar structure, plus `tag2id.json`).

---

## Performing NLU Inference

Use the `NLUInferencer` class from `inference.py` to get predictions on new text.

```python
from inference import NLUInferencer
import json # For pretty printing

# 1. Initialize (loads models from the default './trained_nlu_model' path)
#    This will likely take a few seconds the first time.
try:
    nlu = NLUInferencer()
    print("NLUInferencer initialized successfully.")
except Exception as e:
    print(f"Failed to initialize NLUInferencer: {e}")
    exit()

# 2. Predict on new text
input_text = "My 2022 Ford F-150 needs a tow to Bob's Garage"
try:
    result = nlu.predict(input_text)

    # 3. Analyze the result
    print("\n--- NLU Result ---")
    print(json.dumps(result, indent=2))
    print("------------------")

    intent_name = result.get("intent", {}).get("name", "unknown")
    intent_confidence = result.get("intent", {}).get("confidence", 0.0)
    entities_found = result.get("entities", [])

    print(f"Detected Intent: {intent_name} (Confidence: {intent_confidence:.4f})")
    if entities_found:
        print("Detected Entities:")
        for entity in entities_found:
            print(f"  - {entity['entity']}: {entity['value']}")
    else:
        print("No entities detected.")

except Exception as e:
    print(f"Error during prediction: {e}")

```

**Expected Output Format from `nlu.predict(text)`:**

```json
{
  "text": "My 2022 Ford F-150 needs a tow to Bob's Garage",
  "intent": {
    "name": "towing_request_tow_full", // Predicted intent label
    "confidence": 0.9876 // Model's confidence score (0.0 to 1.0)
  },
  "entities": [
    // List of extracted entities
    {
      "entity": "vehicle_year",
      "value": "2022"
    },
    {
      "entity": "vehicle_make",
      "value": "Ford"
    },
    {
      "entity": "vehicle_model",
      "value": "F-150"
    },
    {
      "entity": "destination",
      "value": "Bob's Garage"
    }
  ]
}
```

---

## Testing

The primary test to verify basic NLU functionality after training is:

````bash
python test_integration.py```

This script uses the `NLUInferencer` to check predictions on a few hardcoded examples. Other `test_phase*.py` scripts exist from previous development phases but are not essential for verifying the current system's basic operation.

---

## CI Workflow

The GitHub Actions workflow in `.github/workflows/ci.yml` automatically runs linters (`flake8`, `black`, `isort`) and the `test_integration.py` script on pushes and pull requests to the `main` and `development` branches.

---

## Technology Stack

*   **Python:** Core language
*   **Hugging Face Transformers:** For DistilBERT models and tokenizers
*   **PyTorch:** Backend for Transformers
*   **Scikit-learn:** For data splitting
*   **Seqeval:** For entity recognition metrics (used during training)
*   **NumPy:** Numerical operations
*   **Black:** Code formatting
````
