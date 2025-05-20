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

## Repository Structure

```
chatbot/
├── src/                       # Core source code
│   ├── api.py                 # API server implementation
│   ├── dialog_manager.py      # Dialog management logic
│   ├── evaluate_nlu.py        # Evaluation and benchmarking utilities
│   ├── inference.py           # Inference utilities
│   ├── model_pipeline.py      # End-to-end model lifecycle management
│   ├── nlu_dashboard.py       # Streamlit dashboard for visualization
│   ├── response_generator.py  # Response generation logic
│   └── train.py               # Model training script
├── scripts/                   # Helper scripts directory
│   ├── run_dashboard.sh       # Script to launch the Streamlit dashboard
│   ├── run_phase5_dashboard.sh # Script for Phase 5 dashboard
│   ├── run_phase4_dashboard.sh # Script for Phase 4 dashboard
│   ├── start_api.sh           # Shell script to start the API server
│   ├── merge_data.py          # Script to merge training data
│   ├── merge_data_unique.py   # Script to merge training data with uniqueness checks
│   ├── merge_data_advanced.py # Advanced script for merging training data
│   ├── analyze_data.py        # Script to analyze training data
│   └── fix_sentiment_data.py  # Script to fix sentiment analysis data
├── tests/                     # Test suite
│   ├── test_api_integration.py # API integration tests
│   ├── test_dialog_manager_unified.py # Tests for dialog manager
│   ├── test_integration.py    # Integration tests
│   ├── test_nlu_regression.py # Regression tests
│   ├── test_phase5.py         # Phase 5 tests
│   └── test_ui_components.py  # UI component tests
├── docs/                      # Documentation
│   ├── API_README.md          # API documentation
│   ├── dashboard_user_guide.md # Dashboard user guide
│   ├── metrics_glossary.md    # Explanation of metrics
│   ├── troubleshooting_guide.md # Troubleshooting help
│   └── nlu_best_practices.md  # Best practices guide
├── config/                    # Configuration files
│   ├── requirements.txt       # Full Python dependencies
│   ├── requirements-api.txt   # API server dependencies
│   ├── requirements-dashboard.txt # Dashboard dependencies
│   ├── regression_config.yml  # Configuration for regression testing
│   └── Dockerfile             # Docker container definition
├── data/                      # Data files
│   └── new_training_data.json # New training data
├── benchmark_results/         # Benchmark results storage
├── trained_nlu_model/         # Trained model storage
├── utils/                     # Utilities and helper functions
├── assets/                    # Static assets for the dashboard
├── pages/                     # Streamlit multi-page components
└── README.md                  # This file
```

---

## Architecture

The system follows a modular architecture with clear separation of concerns:

1. **NLU Layer (`inference.py`):** Handles text processing, intent detection, entity extraction, and sentiment analysis.
2. **Dialog Management Layer (`dialog_manager.py`):** Manages conversation state, determines actions based on intents/entities/sentiment.
3. **Response Generation Layer (`response_generator.py`):** Creates natural language responses based on actions and sentiment context.
4. **API Layer (`api.py`):** Exposes the functionality via a REST API using FastAPI.
5. **Utilities Layer (`utils/`):** Provides helper functions for path handling and other common tasks.

Key architectural features:

- **Dependency Injection:** The DialogManager requires an NLUInferencer instance to be provided at initialization, allowing for loose coupling and easier testing.
- **State Management:** Conversation state is maintained per conversation ID, allowing for multiple simultaneous conversations.
- **Action Determination:** Based on the current state, NLU results, and sentiment analysis, the DialogManager determines appropriate actions.
- **Templated Responses:** The ResponseGenerator uses templates to create varied and context-appropriate responses, with special variations for negative sentiment.
- **Hardware Acceleration:** The system automatically detects and uses Apple Silicon GPU (MPS) for inference and training when available.
- **Client Tracking:** The API server includes client tracking middleware to log connections from different platforms (web, React Native, etc.)

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

This is the **primary** file used for training. It's a JSON list of dictionaries, each representing one training example:

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

The repository also includes utility scripts for merging and analyzing training data:

- `scripts/merge_data.py`: Basic script to merge training data files
- `scripts/merge_data_unique.py`: Merge training data while ensuring uniqueness
- `scripts/merge_data_advanced.py`: Advanced script with more options for merging data
- `scripts/analyze_data.py`: Tool to analyze and report on training data quality

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
7.  **Checkpoint Saving:** Saves checkpoints during training to `./trained_nlu_model/intent_model_checkpoints/` and `./trained_nlu_model/entity_model_checkpoints/`.

---

## NLU Model Benchmarking System

The project includes a comprehensive benchmarking system for evaluating NLU model performance:

### Core Components

1. **Evaluation Script** (`evaluate_nlu.py`):

   - Modular design for efficient NLU model benchmarking
   - Optimized entity handling with BIO tagging
   - Comprehensive metrics calculation (accuracy, F1, precision, recall)
   - Detailed error analysis with pattern detection

2. **Metrics Tracking** (`benchmark_results/metrics_history.csv`):

   - Historical performance tracking over time
   - Efficient file operations with atomic writing
   - Minimal data duplication for better storage efficiency

3. **Visualization System**:

   - Performance trend charts using matplotlib/seaborn
   - Intelligent confusion matrix visualization
   - Class-specific performance analysis
   - Error pattern visualization
   - HTML report generation with embedded charts

4. **Streamlit Dashboard** (`nlu_dashboard.py`):

   - Interactive visualization of benchmark results
   - Historical performance tracking
   - Intent and entity analysis views
   - Error analysis with actionable insights
   - Model comparison capabilities

5. **Regression Testing** (`test_nlu_regression.py`):

   - Statistical significance testing for performance changes
   - Configurable thresholds for different metrics
   - Support for high-impact intent prioritization
   - CI/CD integration for automated quality gates
   - Multiple output formats (text, JSON, GitHub)

6. **Model Pipeline Integration** (`model_pipeline.py`):
   - End-to-end model lifecycle management
   - Versioned model storage with metadata
   - Integrated benchmarking and regression testing
   - Export capabilities for production deployment

### Using the Benchmarking System

#### Basic Evaluation

```bash
# Run a basic evaluation with default settings
python evaluate_nlu.py

# Customize evaluation parameters
python evaluate_nlu.py --benchmark data/nlu_benchmark_data.json --model trained_nlu_model --output benchmark_results
```

#### Running the Dashboard

```bash
# Start the Streamlit dashboard
./scripts/run_dashboard.sh

# The dashboard will be available at http://localhost:8501
```

#### Regression Testing

```bash
# Check if model performance has regressed
python test_nlu_regression.py --metrics-file benchmark_results/metrics_latest.json

# Use custom configuration
python test_nlu_regression.py --config regression_config.yml
```

#### Full Pipeline

```bash
# Run the full model pipeline (train, benchmark, regression test)
python model_pipeline.py pipeline \
  --training-data data/nlu_training_data.json \
  --benchmark-data data/nlu_benchmark_data.json \
  --description "Test pipeline"
```

### Configuration

Custom regression testing thresholds can be defined in `regression_config.yml`:

```yaml
thresholds:
  intent_f1: 0.01 # 1% decrease in intent F1
  entity_f1: 0.02 # 2% decrease in entity F1
  high_impact_intents: 0.03 # 3% decrease for critical intents

high_impact_intents:
  - towing_request_tow_urgent
  - roadside_emergency_situation

significance_level: 0.05 # p-value threshold (95% confidence)
```

### Benchmark Results Storage

Benchmark results are stored in the `benchmark_results` directory:

- `metrics_*.json`: Individual benchmark run results
- `metrics_history.csv`: Historical performance data
- `visualizations/`: Generated charts and visualizations

For more details, see the [NLU Benchmarking Documentation](docs/nlu_benchmarking.md).

---

## Running the API Server

To run the API server:

```bash
# Use port 8001 to avoid conflicts with the Docker container that uses port 8000
# You can also set a custom port using the PORT environment variable
export PORT=8001 && python api.py
```

Alternatively, you can use the provided shell script:

```bash
./scripts/start_api.sh
```

This starts a FastAPI server on http://localhost:8001 that provides:

- `/api/health` - Health check endpoint
- `/api/nlu` - NLU processing endpoint (includes sentiment analysis)
- `/api/dialog` - Dialog processing endpoint that maintains conversation state
- `/api/connections` - Internal endpoint for monitoring active client connections

**IMPORTANT NOTE**: There is a Docker container running on port 8000 with an older version of the API. For development and testing, use the local API on port 8001 as shown above. This ensures you're working with the latest version of the code.

See the `API_README.md` file for detailed API documentation and integration examples.

---

## Docker Support

The project includes Docker support with:

```bash
# Build and start the Docker container
docker-compose up -d
```

This will run the API server inside a containerized environment on port 8000. For most development work, it's recommended to run the server directly on your machine using the instructions above.

---

## Sentiment Analysis

The system now includes sentiment analysis as part of the NLU pipeline:

- **Technical Implementation:** Uses Hugging Face's sentiment analysis pipeline with a pre-trained DistilBERT model.
- **Integration:** Sentiment results (label and score) are included in the NLU output and stored in the dialog state.
- **Dialog Flow Impact:** High negative sentiment combined with certain intents (like towing requests) can trigger expedited "urgent" flows.
- **Response Adaptation:** The response generator uses sentiment information to provide more empathetic, urgent-focused responses for negative sentiment situations.

This feature is particularly valuable for automotive assistance scenarios where user urgency and emotional state can significantly impact the required service level.

---

## Client Platform Detection

The API includes client platform detection middleware that:

1. Automatically identifies if the client is a web browser, React Native app, iOS, or Android
2. Logs connection information for monitoring
3. Adapts response handling based on the client platform when necessary
4. Tracks active connections for system monitoring

To identify your app as a React Native client, add the `X-Platform: React Native` header to your requests.

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

# Additional tests
python test_phase5.py
```

---

## CI Workflow

The GitHub Actions workflow in `.github/workflows/ci.yml` automatically runs linters (`flake8`, `black`, `isort`) and the test scripts on pushes and pull requests to the `main` and `development` branches.

---

## Technology Stack

- **Python:** Core language
- **Hugging Face Transformers:** For DistilBERT models and tokenizers
- **PyTorch:** Backend for Transformers (with MPS support for Apple Silicon)
- **FastAPI:** REST API framework
- **Uvicorn:** ASGI server for FastAPI
- **Scikit-learn:** For data splitting and metrics
- **Seqeval:** For entity recognition metrics
- **NumPy/Pandas:** Data handling and numerical operations
- **Matplotlib/Seaborn:** Visualization libraries
- **Streamlit:** Interactive dashboard framework
- **PyYAML:** Configuration management
- **SciPy:** Statistical significance testing
- **Black:** Code formatting
- **Pytest:** For testing
- **Docker:** For containerization
