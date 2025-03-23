# Implementation Plan for Improved Context Handling

## Problem Overview

- [x] Initial problem assessment and scope definition
- [x] Review of current architecture limitations

The current chatbot is built on a DistilBERT-based architecture for intent classification and entity extraction, but it struggles with negation (e.g., "I don't need a tow truck") and context switching (e.g., "Actually, forget the tow truck; I need a new battery"). This implementation plan will enhance the system to better handle these scenarios through a phased approach with testing at each stage.

## Current Implementation Status

After a thorough review of the existing codebase, we've found that significant progress has already been made toward implementing context-aware functionality:

- [x] Basic `ContextAwareCarroAssistant` class exists, extending the base `CarroAssistant`
- [x] Functions for dataset generation (`generate_negation_dataset`, `generate_context_switch_dataset`)
- [x] Comprehensive test file `test_context_integration.py` with test cases
- [x] Streamlit integration with context visualization and processing modes
- [x] Model loading and context tracking infrastructure

This updated implementation plan will focus on leveraging these existing components while filling implementation gaps and enhancing the context-aware functionality.

## Implementation Phases

### Phase 1: Data Collection and Model Training

**Goal:** Enhance and utilize existing dataset generation functions for negation and context switching models.

#### Tasks:

- [ ] Review and extend existing dataset generation functions
  - [ ] Add additional examples to `generate_negation_dataset` if needed
  - [ ] Enhance `generate_context_switch_dataset` with more varied examples
  - [ ] Verify data augmentation techniques
- [ ] Utilize existing `train_binary_classifier` function
  - [ ] Verify configuration parameters
  - [ ] Run training for both negation and context switch models

#### Testing Checkpoint:

- [ ] Run tests to verify model creation using existing infrastructure
  ```bash
  # Test model creation and basic functionality
  python train_context_models.py
  ```
- [ ] Verify models achieve >85% accuracy on test sets
- [ ] Ensure rule-based fallbacks work when models are unavailable
  ```bash
  python test_context_integration.py TestContextIntegration.test_rule_based_fallbacks
  ```

### Phase 2: Context Tracking Enhancements

**Goal:** Improve existing context tracking infrastructure in the `ContextAwareCarroAssistant` class.

#### Tasks:

- [ ] Review existing context tracking data structures

  - [ ] Verify intent history tracking functionality
  - [ ] Enhance entity tracking with additional metadata
  - [ ] Review conversation state management

- [ ] Extend the existing `ContextAwareCarroAssistant` class
  - [ ] Verify backward compatibility with base `CarroAssistant`
  - [ ] Add additional context-tracking methods if needed
  - [ ] Improve model loading error handling

#### Testing Checkpoint:

- [ ] Run existing tests to verify compatibility
  ```bash
  python test_context_integration.py TestContextIntegration.test_standard_assistant_compatibility
  ```
- [ ] Run existing basic functionality tests
  ```bash
  python test_context_integration.py TestContextIntegration.test_context_assistant_basic
  ```
- [ ] Verify both assistants can process the same inputs

### Phase 3: Detection Method Refinements

**Goal:** Refine existing detection methods for negation, context switching, and contradictions.

#### Tasks:

- [ ] Verify existing negation detection implementation

  - [ ] Test model-based detection with new examples
  - [ ] Enhance rule-based fallback with additional patterns
  - [ ] Improve confidence scoring mechanism

- [ ] Enhance context switch detection

  - [ ] Test existing model-based detection
  - [ ] Add more sophisticated heuristics to rule-based detection
  - [ ] Improve context transition tracking

- [ ] Review contradiction detection
  - [ ] Test entity comparison logic with various entity types
  - [ ] Improve entity history tracking
  - [ ] Add support for partial matches and fuzzy matching

#### Testing Checkpoint:

- [ ] Run existing negation detection tests
  ```bash
  python test_context_integration.py TestContextIntegration.test_negation_detection
  ```
- [ ] Run existing context switch detection tests
  ```bash
  python test_context_integration.py TestContextIntegration.test_context_switch_detection
  ```
- [ ] Run existing contradiction detection tests
  ```bash
  python test_context_integration.py TestContextIntegration.test_contradiction_detection
  ```

### Phase 4: Enhanced Message Processing

**Goal:** Fine-tune existing message processing pipeline for improved context awareness.

#### Tasks:

- [ ] Review existing `process_message_with_context` method

  - [ ] Verify proper integration of all detection mechanisms
  - [ ] Ensure conversation context updates correctly
  - [ ] Optimize processing flow

- [ ] Enhance response adaptation
  - [ ] Create more nuanced responses for negation scenarios
  - [ ] Improve handling of context switches in responses
  - [ ] Add more sophisticated contradiction resolution

#### Testing Checkpoint:

- [ ] Run existing end-to-end processing tests
  ```bash
  python test_context_integration.py TestContextIntegration.test_context_assistant_with_context
  ```
- [ ] Add additional tests for multi-turn conversations
- [ ] Test with sample dialogues to ensure correct behavior

### Phase 5: Streamlit Integration Improvements

**Goal:** Enhance existing Streamlit integration for better context visualization.

#### Tasks:

- [ ] Improve existing session state management

  - [ ] Review context tracking in session state
  - [ ] Enhance conversation history management
  - [ ] Verify context reset functionality

- [ ] Enhance context visualization components
  - [ ] Improve display of active context elements
  - [ ] Add clearer indicators for context switches
  - [ ] Provide better visualization of contradictions

#### Testing Checkpoint:

- [ ] Manual testing of existing UI with sample conversations
  ```bash
  streamlit run streamlit_app.py
  ```
- [ ] Verify improved context indicators appear correctly
- [ ] Test context persistence between page refreshes

### Phase 6: Comprehensive Evaluation

**Goal:** Develop and run comprehensive evaluation metrics for the enhanced context-aware system.

#### Tasks:

- [ ] Extend existing test suite with additional test cases

  - [ ] Add more diverse test cases for each feature
  - [ ] Create additional multi-turn conversation tests
  - [ ] Add more edge cases and challenging scenarios

- [ ] Implement evaluation metrics
  - [ ] Add precision/recall metrics for each detection type
  - [ ] Track conversation-level success rate
  - [ ] Compare with baseline system

#### Testing Checkpoint:

- [ ] Run comprehensive evaluation
  ```bash
  python evaluation.py --mode=context --test_suite=comprehensive
  ```
- [ ] Generate performance report
- [ ] Verify metrics meet target thresholds:
  - [ ] Negation Detection Accuracy: >90%
  - [ ] Context Switch Detection: >85%
  - [ ] Entity Tracking Accuracy: >90%
  - [ ] Conversation Completion Rate: >80%

## Success Metrics

### Core Detection Performance

- [ ] Negation Detection Accuracy: >90%
- [ ] Negation Recovery Rate: >85%
- [ ] Context Switch Detection: >85%
- [ ] Contradiction Rate: <10%

### Conversation Quality

- [ ] Contextual Consistency Score: >85%
- [ ] Conversation Completion Rate: >80%
- [ ] Task Completion Improvement: >20%

### User Experience

- [ ] User Satisfaction Rating: >4/5
- [ ] Clarification Reduction: >30%

## Implementation Recommendations

The following sections contain recommendations for enhancing the existing implementation:

### 1. Model Versioning and Compatibility

Ensure proper model versioning throughout the implementation process:

```python
# In ContextAwareCarroAssistant.__init__
self.version = "context_aware_v1.1"  # Update version number with each significant change

# Document model version compatibility
model_compatibility = {
    "context_aware_v1.0": ["negation_detector_v1", "context_switch_detector_v1"],
    "context_aware_v1.1": ["negation_detector_v1.1", "context_switch_detector_v1.1"]
}
```

### 2. Enhanced Negation Dataset

Consider adding these additional negation patterns to the existing dataset:

```python
# Additional negation examples to consider adding
additional_negation_examples = [
    "I no longer require that service",
    "Let's not do the tow truck after all",
    "I've decided I don't need assistance anymore",
    "On second thought, I don't want roadside help",
    "I'd rather not have a tow truck sent now"
]

# Additional contrastive examples
additional_contrastive_examples = [
    "I need something different than a tow truck",
    "What I actually need is roadside assistance, not a tow",
    "Instead of towing, I need a battery jump"
]
```

### 3. Improved Context Tracking

Enhance the conversation context structure to include additional metadata:

```python
# Enhanced conversation context structure
self.conversation_context = {
    "previous_intents": [],        # Track recent intents
    "previous_entities": {},       # Map of entity_type -> [values]
    "active_flow": None,           # Current conversation flow
    "turn_count": 0,               # Track number of turns
    "context_switches": [],        # Track when context switches occurred
    "negations": [],               # Track when negations occurred
    "contradictions": []           # Track contradictions for analysis
}
```

### 4. Streamlit UX Improvements

```python
# Enhanced context visualization with color coding
def display_enhanced_context_indicators(context):
    """Display improved visual indicators for active context elements."""
    st.sidebar.subheader("Conversation Context",
                        help="Shows the current state of the conversation")

    # Show active flow with better styling
    if context.get("active_flow"):
        st.sidebar.success(f"🔄 Flow: {context['active_flow'].capitalize()}")

    # Show entity history with highlighting for contradictions
    if context.get("previous_entities"):
        with st.sidebar.expander("📋 Active Entities", expanded=True):
            for entity_type, values in context["previous_entities"].items():
                if values:
                    most_recent = max(values, key=lambda x: x["turn"])

                    # Highlight contradicted entities
                    is_contradicted = any(c["entity_type"] == entity_type
                                        for c in context.get("contradictions", []))

                    if is_contradicted:
                        st.markdown(f"**{entity_type.title()}**: "
                                  f"~~{most_recent['prev_value']}~~ → "
                                  f"**{most_recent['value']}** ⚠️")
                    else:
                        st.write(f"**{entity_type.title()}**: {most_recent['value']}")
```
