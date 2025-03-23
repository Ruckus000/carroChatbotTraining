# LangGraph Integration for Context-Aware Chatbot

This directory contains the implementation of LangGraph integration to enhance the existing context-aware chatbot, while preserving the current rule-based methods.

## Progress

### Phase 1: Foundation and Adapter Layer (Completed)

- ✅ **Feature Flag System**: Allows incremental deployment and testing of new features
- ✅ **Adapter Interfaces**: Provides integration with existing models and detection methods
- ✅ **LangGraph State Interface**: Defines the state structure for LangGraph nodes

### Next Steps

The implementation follows a phased approach:

1. Phase 1: Foundation and Adapter Layer ✅
2. Phase 2: Mistral Integration as Enhancement
3. Phase 3: LangGraph Integration for Flow Control
4. Phase 4: Logging, Metrics, and Streamlit Integration
5. Phase 5: Final Integration and Deployment

## Setup

```bash
# Install required dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/test_phase1.py -v
```

## Architecture

The integration uses an adapter pattern to connect the existing rule-based system with LangGraph components:

- **Feature Flags**: Controls which enhancements are active
- **Model Adapters**: Interfaces with existing classification models
- **Detection Adapters**: Interfaces with existing detection methods
- **State Definition**: Defines the conversation state structure for LangGraph

## Usage

The integration is designed to be non-disruptive, allowing the existing system to function while introducing enhancements incrementally through feature flags.

```python
from langgraph_integration import FeatureFlags, ExistingDetectionAdapter

# Initialize components
flags = FeatureFlags()
detector = ExistingDetectionAdapter()

# Optionally enable features
flags.enable("use_langgraph")

# Process message using existing or enhanced system depending on flags
result = detector.process_message("I need a tow truck")
```
