# LangGraph Mistral Integration Chatbot

A sophisticated chatbot implementation that integrates LangGraph for conversation flow control with Mistral's large language model capabilities. This system provides robust intent recognition, context management, and performance optimizations.

## Features

- **Hybrid Intent Detection**: Combines rule-based and ML-based intent detection
- **Advanced Context Management**: Tracks conversation context and handles context switches
- **LangGraph Integration**: Uses LangGraph for conversation flow control
- **Mistral 7B Integration**: Leverages Mistral's language model for natural language processing
- **Performance Optimizations**: CPU-optimized processing for efficient resource usage
- **Streamlit UI**: Clean, modern user interface for chatbot interactions
- **Comprehensive Monitoring**: Structured logging and performance metrics collection
- **Feature Flags**: Granular control over system features and fallbacks

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- Mistral API key (for non-mock usage)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/langgraph-mistral-chatbot.git
cd langgraph-mistral-chatbot
```

2. Run the deployment script to set up the environment:

```bash
./deploy.sh development
```

This will:

- Create a virtual environment
- Install all dependencies
- Generate environment-specific settings
- Run tests
- Start the development server

### Manual Setup

If you prefer to set up manually:

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -e .
```

3. Set environment variables:

```bash
export MISTRAL_API_KEY="your-api-key-here"
```

## Usage

### Starting the Chatbot

After installation, you can run the chatbot with:

```bash
python -m streamlit run langgraph_integration/streamlit_integration.py
```

### Chatbot Configuration

You can configure the chatbot by:

1. **Environment Variables**:

   - `MISTRAL_API_KEY`: Your Mistral API key
   - `ENABLE_USE_LANGGRAPH`: Set to "true" to enable LangGraph integration
   - `ENABLE_USE_MISTRAL`: Set to "true" to enable Mistral integration

2. **Configuration Files**:
   - Create `config/deployment.{environment}.yaml` files with settings

### Development

#### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific phase tests
python -m pytest tests/test_phase1.py -v
python -m pytest tests/test_phase2.py -v
python -m pytest tests/test_phase3.py -v
python -m pytest tests/test_phase4.py -v
python -m pytest tests/test_phase5.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v
```

#### Feature Flags

Feature flags allow you to enable or disable specific features:

- `use_langgraph`: Enables the LangGraph workflow for conversation control
- `use_mistral`: Enables Mistral AI integration
- `enable_monitoring`: Enables logging and metrics collection
- `enable_cpu_optimizations`: Enables CPU optimizations for better performance

## Architecture

The system is built with a modular architecture:

1. **Core Components**:

   - `adapters.py`: Base adapter interface
   - `feature_flags.py`: Feature flag configuration
   - `langgraph_state.py`: State definitions for LangGraph

2. **ML Integration**:

   - `mistral_integration.py`: Mistral AI integration
   - `hybrid_detection.py`: Hybrid rule-based and ML-based detection

3. **LangGraph Components**:

   - `langgraph_nodes.py`: Node definitions for the LangGraph workflow
   - `state_converter.py`: Converts between different state representations
   - `langgraph_workflow.py`: Main workflow implementation

4. **Performance & Monitoring**:

   - `monitoring.py`: Metrics collection and logging
   - `cpu_optimizations.py`: CPU-optimized routines

5. **UI Integration**:
   - `streamlit_integration.py`: Streamlit UI implementation

## Deployment

For production deployment:

```bash
./deploy.sh production
```

This script:

- Runs critical tests
- Sets up appropriate environment settings
- Deploys the application as a background service
- Saves logs to the `logs` directory

## License

[MIT License](LICENSE)
