# Implementation Plan for Improved Context Handling

## Problem Overview

Your chatbot is currently built on a DistilBERT-based architecture for intent classification and entity extraction, but it struggles with negation (e.g., "I don't need a tow truck") and context switching (e.g., "Actually, forget the tow truck; I need a new battery"). This implementation plan will enhance your system to better handle these scenarios.

## Improvement Recommendations

After reviewing the codebase and this implementation plan, we've identified several areas that should be addressed to ensure a robust implementation:

### 1. Context Management Enhancements

- **Broader Entity Context**: Store more than just the last 3 intents - entity values are equally important for context understanding and contradiction detection.
- **Conversation State Tracking**: Implement a more robust conversation state tracker that maintains the full context of entities mentioned across turns.

### 2. Model Architecture Improvements

- **Cross-Turn Attention**: Add an attention mechanism between previous conversation turns and current input to better capture context relationships.
- **Refined Negation Detection**: The current plan uses `hidden_states[-1]` for negation detection, but multiple transformer layers or specialized attention mechanisms would be more effective.
- **Alternative Architecture Option**: Consider a modular approach with separate models sharing embeddings as a fallback if the multi-task approach becomes too complex.

### 3. Evaluation and Testing Enhancements

- **Conversation-Level Metrics**: Add full conversation-level success metrics beyond single-turn accuracy.
- **User Simulator Testing**: Implement automated user simulators that can evaluate complex multi-turn scenarios.
- **Production Monitoring**: Add metrics to track context-related failures in production.

### 4. Streamlit Integration Considerations

- **Session Management**: The current Streamlit implementation needs explicit updates to maintain conversation context between turns.
- **UI Feedback**: Add UI elements to show when context is being carried over from previous turns.

These improvements will be integrated into the implementation plan below.

## Step 1: Data Collection and Augmentation

### 1.1 Create New Negation Examples

First, let's create a new file with examples specifically focused on negation and context switching:

```python
# File: chatbot/data/negation_examples.json
[
  {
    "flow": "clarification",
    "intent": "negation_tow",
    "input": "I don't need a tow truck.",
    "response": "I understand you don't need a tow truck. What service can I help you with instead?",
    "context": {"negation": true},
    "entities": []
  },
  {
    "flow": "clarification",
    "intent": "negation_tow_alternative",
    "input": "I don't need a tow truck, I need a tire change.",
    "response": "I understand you need a tire change rather than a tow truck. I can arrange roadside assistance for your tire. Where are you located?",
    "context": {"negation": true, "alternative_requested": true},
    "entities": [
      {"entity": "service_type", "value": "tire change"}
    ]
  },
  {
    "flow": "clarification",
    "intent": "context_switch_tow_to_battery",
    "input": "Actually, forget the tow truck; I need a new battery.",
    "response": "I understand you now need battery service rather than a tow. I'll arrange roadside assistance to bring a new battery. Where are you located?",
    "context": {"context_switch": true, "previous_intent": "tow"},
    "entities": [
      {"entity": "service_type", "value": "battery"}
    ]
  },
  {
    "flow": "clarification",
    "intent": "negation_appointment",
    "input": "I don't want to schedule service now.",
    "response": "I understand you don't want to schedule service at this time. Is there something else I can help you with?",
    "context": {"negation": true},
    "entities": []
  },
  {
    "flow": "clarification",
    "intent": "context_switch_appointment_to_roadside",
    "input": "Actually, I have a more urgent problem. My car won't start.",
    "response": "I understand you have an urgent situation with your car not starting. Let me help you with roadside assistance instead. Could you provide your location?",
    "context": {"context_switch": true, "urgency": "high"},
    "entities": [
      {"entity": "issue_type", "value": "won't start"}
    ]
  }
]
```

### 1.2 Enhance Data Augmentation for Negation

Extend your existing `data_augmentation.py` to include negation patterns and context switching:

```python
def generate_negation_variations(input_text: str, intent: str, flow: str) -> List[str]:
    """Generate variations that include negation patterns"""

    variations = []

    if "tow" in input_text.lower():
        variations.extend([
            f"I don't need {input_text.lower()}",
            f"Actually, forget about {input_text.lower()}",
            f"I no longer need {input_text.lower()}",
            f"I changed my mind about {input_text.lower()}",
            f"I'm not looking for {input_text.lower()} anymore"
        ])
    elif "appointment" in input_text.lower() or "schedule" in input_text.lower():
        variations.extend([
            f"I don't want to {input_text.lower()}",
            f"I've decided against {input_text.lower()}",
            f"Let's cancel {input_text.lower()}",
            f"I'm not interested in {input_text.lower()} anymore",
            f"I won't be {input_text.lower()}"
        ])

    return variations

def generate_context_switch_variations(input_text: str, intent: str, flow: str) -> List[Dict[str, Any]]:
    """Generate examples where the context switches between intents"""

    variations = []

    if flow == "towing":
        variations.extend([
            {
                "flow": "clarification",
                "intent": "context_switch_tow_to_roadside",
                "input": f"Actually, instead of {input_text.lower()}, I just need a jump start.",
                "response": "I understand you need a jump start rather than a tow. I can arrange that. Where are you located?",
                "context": {"context_switch": True, "previous_intent": "tow"},
                "entities": [{"entity": "service_type", "value": "jump start"}]
            },
            {
                "flow": "clarification",
                "intent": "context_switch_tow_to_tire",
                "input": f"On second thought, I don't need {input_text.lower()}. I just need help changing a tire.",
                "response": "I understand you need help with a tire change rather than a tow. I can arrange roadside assistance for that. Where are you located?",
                "context": {"context_switch": True, "previous_intent": "tow"},
                "entities": [{"entity": "service_type", "value": "tire change"}]
            }
        ])
    elif flow == "roadside":
        variations.extend([
            {
                "flow": "clarification",
                "intent": "context_switch_roadside_to_tow",
                "input": f"Actually, {input_text.lower()} won't work. I need a tow truck instead.",
                "response": "I understand you now need a tow truck instead. I can arrange that. Where is your vehicle located and where would you like it towed?",
                "context": {"context_switch": True, "previous_intent": "roadside"},
                "entities": []
            },
            {
                "flow": "clarification",
                "intent": "context_switch_roadside_to_appointment",
                "input": f"Actually, {input_text.lower()} isn't what I need. I'd like to schedule a service appointment.",
                "response": "I understand you'd like to schedule a service appointment instead. I can help with that. What service do you need and when would you like to schedule it?",
                "context": {"context_switch": True, "previous_intent": "roadside"},
                "entities": []
            }
        ])

    return variations
```

### 1.3 Update Augmentation Process

Extend the `augment_conversation_data` function in `data_augmentation.py` to include these new variation generators:

```python
def augment_conversation_data(conversations: List[Dict[str, Any]],
                             extreme_test: bool = False) -> List[Dict[str, Any]]:
    """
    Apply comprehensive data augmentation to conversation data.

    Args:
        conversations: List of conversation dictionaries
        extreme_test: Whether to generate extreme test cases

    Returns:
        List of augmented conversation dictionaries
    """
    conversation_extensions = []

    # Add variations for each conversation
    for example in conversations:
        # Generate standard variations
        variations = generate_variations(example['input'], example['intent'], example['flow'])

        for var in variations:
            new_example = copy.deepcopy(example)
            new_example['input'] = var
            conversation_extensions.append(new_example)

        # Generate negation variations
        negation_variations = generate_negation_variations(example['input'], example['intent'], example['flow'])
        for var in negation_variations:
            new_example = copy.deepcopy(example)
            new_example['input'] = var
            new_example['flow'] = 'clarification'
            new_example['intent'] = f"negation_{example['intent']}"
            new_example['context'] = new_example.get('context', {})
            new_example['context']['negation'] = True
            conversation_extensions.append(new_example)

        # Generate context switch variations
        context_switch_variations = generate_context_switch_variations(example['input'], example['intent'], example['flow'])
        conversation_extensions.extend(context_switch_variations)

        # Create entity variations
        entity_variations = create_entity_variations(example)
        conversation_extensions.extend(entity_variations)

    # Add mixed intent examples
    conversation_extensions.extend(generate_mixed_intent_examples())

    # Add extreme test cases if requested
    if extreme_test:
        conversation_extensions.extend(generate_extreme_test_cases())

    # Combine original conversations with extensions
    all_conversations = conversations + conversation_extensions

    # Apply noise to create more variations
    noisy_conversations = []
    for conv in all_conversations:
        # Only apply to a subset to maintain clean examples as well
        if random.random() < 0.4:  # 40% chance to create a noisy version
            noisy_conv = copy.deepcopy(conv)
            noisy_conv['input'] = add_advanced_noise(conv['input'])
            if noisy_conv['input'] != conv['input']:  # Only add if it's different
                noisy_conversations.append(noisy_conv)

    # Add the noisy conversations to our dataset
    all_conversations.extend(noisy_conversations)

    return all_conversations
```

## Step 2: Update the Training Dataset Structure

### 2.1 Create Clarification Classification Dataset

We need to improve the clarification detection model to recognize context switching and negation:

```python
def create_clarification_dataset(conversations: List[Dict[str, Any]],
                                negation_examples: List[Dict[str, Any]],
                                output_dir: str) -> None:
    """
    Create specialized datasets for clarification detection.

    Args:
        conversations: Regular conversation examples
        negation_examples: Specialized negation and context switch examples
        output_dir: Directory to save the datasets
    """
    clarification_examples = []

    # Extract clarification examples from regular conversations
    for conv in conversations:
        if conv.get('flow') == 'clarification':
            clarification_examples.append({
                "text": conv.get('input', ''),
                "label": "clarification",
                "type": "standard"
            })
        else:
            clarification_examples.append({
                "text": conv.get('input', ''),
                "label": "no_clarification",
                "type": "standard"
            })

    # Add negation examples
    for ex in negation_examples:
        context_dict = ex.get('context', {})
        if isinstance(context_dict, dict) and (context_dict.get('negation', False) or context_dict.get('context_switch', False)):
            label = "clarification"
            if context_dict.get('negation', False):
                type_label = "negation"
            elif context_dict.get('context_switch', False):
                type_label = "context_switch"
            else:
                type_label = "standard"
        else:
            label = "no_clarification"
            type_label = "standard"

        clarification_examples.append({
            "text": ex.get('input', ''),
            "label": label,
            "type": type_label
        })

    # Split into train/val/test
    random.shuffle(clarification_examples)
    split1 = int(len(clarification_examples) * 0.7)
    split2 = int(len(clarification_examples) * 0.85)

    train_examples = clarification_examples[:split1]
    val_examples = clarification_examples[split1:split2]
    test_examples = clarification_examples[split2:]

    # Save datasets
    with open(os.path.join(output_dir, 'clarification_classification_train.json'), 'w') as f:
        json.dump(train_examples, f, indent=2)

    with open(os.path.join(output_dir, 'clarification_classification_val.json'), 'w') as f:
        json.dump(val_examples, f, indent=2)

    with open(os.path.join(output_dir, 'clarification_classification_test.json'), 'w') as f:
        json.dump(test_examples, f, indent=2)
```

## Step 3: Enhance the Model Architecture

### 3.1 Create a Contextual Intent Classifier

Modify your model_training.py to include context sensitivity in the intent classifier with enhanced attention mechanisms:

```python
class ContextualIntentClassifier(DistilBertForSequenceClassification):
    """DistilBERT model with enhanced context awareness for intent classification"""

    def __init__(self, config):
        super().__init__(config)

        # Add a component to handle negation features with multi-layer processing
        self.negation_detector = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size // 2, 1)
        )

        # Add context attention component for cross-turn awareness
        self.context_attention = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1
        )

        # Previous context projection layer
        self.context_projection = torch.nn.Linear(config.hidden_size, config.hidden_size)

        # Register buffer for storing previous turn embeddings
        self.register_buffer("previous_turn_embedding", torch.zeros(1, config.hidden_size))
        self.has_previous_context = False

    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, previous_context=None):
        # Get standard outputs from DistilBERT
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states for context processing
            return_dict=True  # Always return dict for consistency
        )

        # Process hidden states from multiple layers for better feature extraction
        # Use weighted combination of last 3 layers instead of just the last one
        last_hidden_states = outputs.hidden_states[-3:]  # Get last 3 layers
        layer_weights = torch.tensor([0.1, 0.3, 0.6]).to(input_ids.device)  # Weighted importance

        # Apply weighted combination
        weighted_hidden_states = torch.zeros_like(last_hidden_states[-1])
        for i, layer in enumerate(last_hidden_states):
            weighted_hidden_states += layer_weights[i] * layer

        # Get CLS token representation
        pooled_output = weighted_hidden_states[:, 0]  # Use CLS token

        # Apply context attention if previous context exists
        if previous_context is not None:
            # Process previous context
            context_embedding = self.context_projection(previous_context)

            # Reshape for attention
            query = pooled_output.unsqueeze(0)
            key = context_embedding.unsqueeze(0)
            value = context_embedding.unsqueeze(0)

            # Apply cross-turn attention
            context_aware_output, _ = self.context_attention(query, key, value)
            context_aware_output = context_aware_output.squeeze(0)

            # Combine current representation with context-aware representation
            pooled_output = pooled_output + 0.3 * context_aware_output

        # Store current embedding for next turn
        self.previous_turn_embedding = pooled_output.detach()
        self.has_previous_context = True

        # Enhanced negation detection with multi-layer processing
        negation_logits = self.negation_detector(pooled_output)

        # Add negation prediction to outputs
        outputs.negation_logits = negation_logits

        # Store context-aware token representation for downstream tasks
        outputs.context_aware_embedding = pooled_output

        return outputs

def train_contextual_intent_classifier(flow: str, dataset_dir: str, output_dir: str):
    """
    Train an enhanced intent classifier that understands context.

    Args:
        flow: Flow name (e.g., "towing")
        dataset_dir: Directory containing datasets
        output_dir: Directory to save model
    """
    # Implementation similar to train_intent_classifier but using the enhanced model
    logger.info(f"Training contextual intent classifier for {flow} flow")

    # Load datasets
    train_file = os.path.join(dataset_dir, 'intent_classification_train.json')
    val_file = os.path.join(dataset_dir, 'intent_classification_val.json')

    with open(train_file, 'r') as f:
        intent_train_data = json.load(f)

    with open(val_file, 'r') as f:
        intent_val_data = json.load(f)

    # Filter data for the specific flow
    flow_train_data = [x for x in intent_train_data if x.get('flow') == flow]
    flow_val_data = [x for x in intent_val_data if x.get('flow') == flow]

    # Get unique intent labels
    train_labels = set(item['label'] for item in flow_train_data)
    val_labels = set(item['label'] for item in flow_val_data)
    all_labels = sorted(list(train_labels.union(val_labels)))

    # Create label to id mapping
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for i, label in enumerate(all_labels)}

    # Convert data to expected format
    train_texts = [x['text'] for x in flow_train_data]
    train_labels = [label2id[x['label']] for x in flow_train_data]

    val_texts = [x['text'] for x in flow_val_data]
    val_labels = [label2id[x['label']] if x['label'] in label2id else 0 for x in flow_val_data]

    # Enhanced feature extraction for negation and context
    train_features = []
    for x in flow_train_data:
        features = {
            "contains_negation": 1 if any(neg in x['text'].lower() for neg in
                                         ["don't", "not ", "no longer", "forget", "isn't", "aren't", "wasn't"]) else 0,
            "contains_context_switch": 1 if any(cs in x['text'].lower() for cs in
                                              ["actually", "instead", "rather", "changed my mind", "forget"]) else 0,
            "text_length": len(x['text'].split()),
            "has_question": 1 if "?" in x['text'] else 0
        }
        train_features.append(features)

    val_features = []
    for x in flow_val_data:
        features = {
            "contains_negation": 1 if any(neg in x['text'].lower() for neg in
                                         ["don't", "not ", "no longer", "forget", "isn't", "aren't", "wasn't"]) else 0,
            "contains_context_switch": 1 if any(cs in x['text'].lower() for cs in
                                              ["actually", "instead", "rather", "changed my mind", "forget"]) else 0,
            "text_length": len(x['text'].split()),
            "has_question": 1 if "?" in x['text'] else 0
        }
        val_features.append(features)

    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Add extracted features to encodings for custom dataset
    for i, features in enumerate(train_features):
        for feat_name, feat_value in features.items():
            if feat_name not in train_encodings:
                train_encodings[feat_name] = []
            train_encodings[feat_name].append(feat_value)

    for i, features in enumerate(val_features):
        for feat_name, feat_value in features.items():
            if feat_name not in val_encodings:
                val_encodings[feat_name] = []
            val_encodings[feat_name].append(feat_value)

    # Create custom dataset to handle additional features
    class ContextualIntentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) if not isinstance(val[idx], int) else torch.tensor([val[idx]])
                   for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # Create datasets
    train_dataset = ContextualIntentDataset(train_encodings, train_labels)
    val_dataset = ContextualIntentDataset(val_encodings, val_labels)

    # Initialize model with context awareness
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        output_hidden_states=True  # Need hidden states for negation detection
    )

    model = ContextualIntentClassifier.from_pretrained(
        'distilbert-base-uncased',
        config=config
    )

    # Training code continues as in original train_intent_classifier
    # ...
```

### 3.2 Implement Multi-task Learning for Clarification Detection

Update your training approach to incorporate multi-task learning with improved architecture:

```python
def train_multi_task_clarification_classifier(dataset_dir: str, output_dir: str):
    """
    Train an enhanced multi-task classifier that detects clarification needs, negation,
    and context switching with shared representations.

    Args:
        dataset_dir: Directory containing datasets
        output_dir: Directory to save model
    """
    logger.info("Training multi-task clarification classifier with enhanced architecture")

    # Load datasets
    train_file = os.path.join(dataset_dir, 'clarification_classification_train.json')
    val_file = os.path.join(dataset_dir, 'clarification_classification_val.json')

    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(val_file, 'r') as f:
        val_data = json.load(f)

    # Log dataset statistics for monitoring class imbalance
    clarification_count = sum(1 for item in train_data if item['label'] == 'clarification')
    negation_count = sum(1 for item in train_data if item.get('type') == 'negation')
    context_switch_count = sum(1 for item in train_data if item.get('type') == 'context_switch')

    logger.info(f"Training data statistics:")
    logger.info(f"  Total examples: {len(train_data)}")
    logger.info(f"  Clarification examples: {clarification_count} ({clarification_count/len(train_data)*100:.1f}%)")
    logger.info(f"  Negation examples: {negation_count} ({negation_count/len(train_data)*100:.1f}%)")
    logger.info(f"  Context switch examples: {context_switch_count} ({context_switch_count/len(train_data)*100:.1f}%)")

    # Prepare multi-task labels
    train_texts = [item['text'] for item in train_data]
    train_clarification_labels = [1 if item['label'] == 'clarification' else 0 for item in train_data]
    train_negation_labels = [1 if item.get('type') == 'negation' else 0 for item in train_data]
    train_context_switch_labels = [1 if item.get('type') == 'context_switch' else 0 for item in train_data]

    val_texts = [item['text'] for item in val_data]
    val_clarification_labels = [1 if item['label'] == 'clarification' else 0 for item in val_data]
    val_negation_labels = [1 if item.get('type') == 'negation' else 0 for item in val_data]
    val_context_switch_labels = [1 if item.get('type') == 'context_switch' else 0 for item in val_data]

    # Create a custom dataset for multi-task learning with improved features
    class MultiTaskDataset(Dataset):
        def __init__(self, encodings, clarification_labels, negation_labels, context_switch_labels):
            self.encodings = encodings
            self.clarification_labels = clarification_labels
            self.negation_labels = negation_labels
            self.context_switch_labels = context_switch_labels

            # Calculate class weights for balanced loss
            pos_weight_clarification = (len(clarification_labels) - sum(clarification_labels)) / max(1, sum(clarification_labels))
            pos_weight_negation = (len(negation_labels) - sum(negation_labels)) / max(1, sum(negation_labels))
            pos_weight_context_switch = (len(context_switch_labels) - sum(context_switch_labels)) / max(1, sum(context_switch_labels))

            self.class_weights = {
                "clarification": torch.tensor([1.0, pos_weight_clarification]),
                "negation": torch.tensor([1.0, pos_weight_negation]),
                "context_switch": torch.tensor([1.0, pos_weight_context_switch])
            }

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['clarification_labels'] = torch.tensor(self.clarification_labels[idx])
            item['negation_labels'] = torch.tensor(self.negation_labels[idx])
            item['context_switch_labels'] = torch.tensor(self.context_switch_labels[idx])
            return item

        def __len__(self):
            return len(self.clarification_labels)

        def get_class_weights(self):
            return self.class_weights

    # Extract additional linguistic features for improved detection
    def extract_linguistic_features(texts):
        features = []
        for text in texts:
            # Check for negation indicators
            contains_negation = int(any(neg in text.lower() for neg in
                                     ["don't", "not ", "no longer", "forget", "isn't", "aren't", "wasn't"]))

            # Check for context switch indicators
            contains_switch = int(any(cs in text.lower() for cs in
                                    ["actually", "instead", "rather", "changed my mind", "forget"]))

            # Check for question indicators
            contains_question = int("?" in text)

            # Check for uncertainty indicators
            contains_uncertainty = int(any(unc in text.lower() for neg in
                                         ["maybe", "perhaps", "possibly", "not sure", "might", "could"]))

            # Text length as proxy for complexity
            text_length = min(50, len(text.split())) / 50.0  # Normalize to 0-1

            features.append({
                "negation_feature": contains_negation,
                "switch_feature": contains_switch,
                "question_feature": contains_question,
                "uncertainty_feature": contains_uncertainty,
                "length_feature": text_length
            })
        return features

    # Extract linguistic features
    train_linguistic_features = extract_linguistic_features(train_texts)
    val_linguistic_features = extract_linguistic_features(val_texts)

    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Add linguistic features to encodings
    for feature_name in train_linguistic_features[0].keys():
        train_encodings[feature_name] = [item[feature_name] for item in train_linguistic_features]
        val_encodings[feature_name] = [item[feature_name] for item in val_linguistic_features]

    # Create datasets
    train_dataset = MultiTaskDataset(train_encodings, train_clarification_labels, train_negation_labels, train_context_switch_labels)
    val_dataset = MultiTaskDataset(val_encodings, val_clarification_labels, val_negation_labels, val_context_switch_labels)

    # Get class weights for balanced training
    class_weights = train_dataset.get_class_weights()

    # Create a custom multi-task model with enhanced architecture
    class EnhancedMultiTaskModel(DistilBertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.distilbert = DistilBertModel(config)

            # Feature fusion layer - combine transformer outputs with linguistic features
            self.feature_dim = config.dim + 5  # 5 linguistic features

            # Enhanced shared representation layer
            self.shared_layer = torch.nn.Sequential(
                torch.nn.Linear(self.feature_dim, self.feature_dim),
                torch.nn.LayerNorm(self.feature_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
            )

            # Task-specific representation layers for better specialization
            self.clarification_representation = torch.nn.Sequential(
                torch.nn.Linear(self.feature_dim, config.dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )

            self.negation_representation = torch.nn.Sequential(
                torch.nn.Linear(self.feature_dim, config.dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )

            self.context_switch_representation = torch.nn.Sequential(
                torch.nn.Linear(self.feature_dim, config.dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )

            # Classification heads
            self.clarification_classifier = torch.nn.Linear(config.dim, 2)
            self.negation_classifier = torch.nn.Linear(config.dim, 2)
            self.context_switch_classifier = torch.nn.Linear(config.dim, 2)

            # Apply weight initialization
            self.init_weights()

        def forward(self, input_ids=None, attention_mask=None,
                  clarification_labels=None, negation_labels=None, context_switch_labels=None,
                  negation_feature=None, switch_feature=None, question_feature=None,
                  uncertainty_feature=None, length_feature=None):
            # DistilBERT forward pass
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Extract sequence output and get CLS token
            hidden_state = outputs[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)

            # Prepare linguistic features tensor
            linguistic_features = torch.stack([
                negation_feature, switch_feature, question_feature,
                uncertainty_feature, length_feature
            ], dim=1).float()

            # Combine transformer output with linguistic features
            combined_features = torch.cat([pooled_output, linguistic_features], dim=1)

            # Apply shared representation layer
            shared_representation = self.shared_layer(combined_features)

            # Apply task-specific representations
            clarification_representation = self.clarification_representation(shared_representation)
            negation_representation = self.negation_representation(shared_representation)
            context_switch_representation = self.context_switch_representation(shared_representation)

            # Generate logits for each task
            clarification_logits = self.clarification_classifier(clarification_representation)
            negation_logits = self.negation_classifier(negation_representation)
            context_switch_logits = self.context_switch_classifier(context_switch_representation)

            # Calculate loss if labels are provided
            loss = None
            if clarification_labels is not None and negation_labels is not None and context_switch_labels is not None:
                # Use weighted cross entropy loss for handling class imbalance
                loss_fct = torch.nn.CrossEntropyLoss(weight=None)  # Weights handled in training loop

                clarification_loss = loss_fct(clarification_logits.view(-1, 2), clarification_labels.view(-1))
                negation_loss = loss_fct(negation_logits.view(-1, 2), negation_labels.view(-1))
                context_switch_loss = loss_fct(context_switch_logits.view(-1, 2), context_switch_labels.view(-1))

                # Combined loss with adjustable weights
                # Prioritize tasks based on their importance
                loss = clarification_loss + 0.7 * negation_loss + 0.7 * context_switch_loss

            return {
                'loss': loss,
                'clarification_logits': clarification_logits,
                'negation_logits': negation_logits,
                'context_switch_logits': context_switch_logits,
                'shared_representation': shared_representation
            }

    # Initialize model
    model = EnhancedMultiTaskModel.from_pretrained('distilbert-base-uncased')

    # Define training arguments with early stopping and learning rate scheduling
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        no_cuda=True  # Using CPU
    )

    # Custom compute_metrics function to track performance
    def compute_metrics(eval_pred):
        results = {}

        # Unpack predictions and labels
        predictions = eval_pred.predictions
        clarification_preds = np.argmax(predictions[0], axis=1)
        negation_preds = np.argmax(predictions[1], axis=1)
        context_switch_preds = np.argmax(predictions[2], axis=1)

        clarification_labels = eval_pred.label_ids[0]
        negation_labels = eval_pred.label_ids[1]
        context_switch_labels = eval_pred.label_ids[2]

        # Calculate metrics for each task
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

        # Clarification metrics
        results["clarification_accuracy"] = accuracy_score(clarification_labels, clarification_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            clarification_labels, clarification_preds, average="binary"
        )
        results["clarification_precision"] = precision
        results["clarification_recall"] = recall
        results["clarification_f1"] = f1

        # Negation metrics
        results["negation_accuracy"] = accuracy_score(negation_labels, negation_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            negation_labels, negation_preds, average="binary"
        )
        results["negation_precision"] = precision
        results["negation_recall"] = recall
        results["negation_f1"] = f1

        # Context switch metrics
        results["context_switch_accuracy"] = accuracy_score(context_switch_labels, context_switch_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            context_switch_labels, context_switch_preds, average="binary"
        )
        results["context_switch_precision"] = precision
        results["context_switch_recall"] = recall
        results["context_switch_f1"] = f1

        # Calculate macro-averaged F1 across all tasks
        results["f1_macro"] = (results["clarification_f1"] + results["negation_f1"] + results["context_switch_f1"]) / 3

        return results

    # Initialize the CustomTrainer with our multi-task setup
    class MultiTaskTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = {
                "clarification_labels": inputs.pop("clarification_labels"),
                "negation_labels": inputs.pop("negation_labels"),
                "context_switch_labels": inputs.pop("context_switch_labels")
            }

            outputs = model(**inputs, **labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on validation set
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Save the model
    model_path = os.path.join(output_dir, "multi_task_clarification")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Save task-specific class weights for inference
    with open(os.path.join(model_path, "class_weights.json"), "w") as f:
        json.dump({
            "clarification": class_weights["clarification"].tolist(),
            "negation": class_weights["negation"].tolist(),
            "context_switch": class_weights["context_switch"].tolist()
        }, f)

    # Save evaluation results
    with open(os.path.join(model_path, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)

    logger.info(f"Multi-task clarification model saved to {model_path}")

    return model_path
```

## Step 4: Implement Clarification Mechanism

### 4.1 Update the Inference Process

Enhance the `inference.py` file to incorporate context awareness in the inference process:

````python
class ContextAwareCarroAssistant:
    def __init__(self, models_dir):
        """
        Initialize with enhanced context-aware models.

        Args:
            models_dir: Directory containing the trained models
        """
        # (Load models as in original CarroAssistant)

        # Load the multi-task clarification model
        clarification_model_path = os.path.join(models_dir, "multi_task_clarification")
        self.clarification_tokenizer = DistilBertTokenizer.from_pretrained(clarification_model_path)
        self.clarification_model = MultiTaskClarificationModel.from_pretrained(clarification_model_path)
        self.clarification_model.to(device)
        self.clarification_model.eval()

        # Maintain conversation context - ENHANCED CONTEXT TRACKING
        self.conversation_context = {
            "previous_intents": [],        # Track last 5 intents
            "previous_entities": [],       # Track all entities from conversation
            "entity_history": {},          # Map of entity_type -> [values across turns]
            "active_flow": None,           # Current flow
            "previous_flows": [],          # Track flow transitions
            "session_start_time": None,    # Track session duration
            "contradiction_history": [],   # Track detected contradictions
            "turn_count": 0                # Track conversation length
        }

        # Configure context tracking parameters
        self.context_config = {
            "max_intents_history": 5,      # Maximum intents to track (increased from 3)
            "max_turns_for_context": 10,   # Maximum conversation turns to consider
            "contradiction_threshold": 0.7  # Confidence threshold for contradiction detection
        }

    def detect_context_features(self, text):
        """
        Detect negation, context switching and clarification needs.

        Args:
            text: User input text

        Returns:
            Dictionary with detection results
        """
        inputs = self.clarification_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clarification_model(**inputs)

        # Get predictions from each classification head
        clarification_logits = outputs['clarification_logits']
        negation_logits = outputs['negation_logits']
        context_switch_logits = outputs['context_switch_logits']

        clarification_score = torch.softmax(clarification_logits, dim=1)[:, 1].item()
        negation_score = torch.softmax(negation_logits, dim=1)[:, 1].item()
        context_switch_score = torch.softmax(context_switch_logits, dim=1)[:, 1].item()

        # Apply thresholds
        needs_clarification = clarification_score > 0.5
        contains_negation = negation_score > 0.5
        contains_context_switch = context_switch_score > 0.5

        return {
            'needs_clarification': needs_clarification,
            'contains_negation': contains_negation,
            'contains_context_switch': contains_context_switch,
            'clarification_score': clarification_score,
            'negation_score': negation_score,
            'context_switch_score': context_switch_score
        }

    def update_conversation_context(self, processing_result):
        """
        Update the conversation context based on latest processing result.
        Enhanced with more robust entity tracking and context management.

        Args:
            processing_result: Result from process_message
        """
        # Increment turn count
        self.conversation_context["turn_count"] += 1

        # Store previous intent in context
        if processing_result['intent'] not in ['unknown', 'clarification_needed']:
            self.conversation_context['previous_intents'].append({
                'intent': processing_result['intent'],
                'turn': self.conversation_context["turn_count"],
                'confidence': processing_result.get('intent_confidence', 1.0)
            })

            # Keep limited history but more than before
            if len(self.conversation_context['previous_intents']) > self.context_config["max_intents_history"]:
                self.conversation_context['previous_intents'].pop(0)

        # Enhanced entity tracking with timestamps and confidence
        for entity_type, values in processing_result['entities'].items():
            # Initialize entity type history if not exists
            if entity_type not in self.conversation_context['entity_history']:
                self.conversation_context['entity_history'][entity_type] = []

            # Add each entity value with metadata
            for value in values:
                entity_entry = {
                    'entity': entity_type,
                    'value': value,
                    'turn': self.conversation_context["turn_count"],
                    'confidence': processing_result.get('entity_confidence', {}).get(entity_type, 1.0)
                }

                # Add to general entities list
                self.conversation_context['previous_entities'].append(entity_entry)

                # Add to type-specific history
                self.conversation_context['entity_history'][entity_type].append({
                    'value': value,
                    'turn': self.conversation_context["turn_count"],
                    'confidence': processing_result.get('entity_confidence', {}).get(entity_type, 1.0)
                })

        # Update active flow and track flow transitions
        if processing_result.get('flow') and processing_result['flow'] not in ['clarification', 'fallback']:
            # If flow changed, record the transition
            if self.conversation_context['active_flow'] and self.conversation_context['active_flow'] != processing_result['flow']:
                self.conversation_context['previous_flows'].append({
                    'from': self.conversation_context['active_flow'],
                    'to': processing_result['flow'],
                    'turn': self.conversation_context["turn_count"]
                })

            # Update current flow
            self.conversation_context['active_flow'] = processing_result['flow']

    def process_message(self, text):
        """
        Process an incoming message with context awareness.

        Args:
            text: User input text

        Returns:
            Dict containing intent, entities, and context information
        """
        # First check for context-related features
        context_features = self.detect_context_features(text)

        # Initialize result
        result = {
            'intent': 'unknown',
            'entities': {},
            'needs_fallback': False,
            'needs_clarification': context_features['needs_clarification'],
            'contains_negation': context_features['contains_negation'],
            'contains_context_switch': context_features['contains_context_switch']
        }

        # Handle negation
        if context_features['contains_negation']:
            # Check what's being negated - analysis of previous context
            if self.conversation_context['previous_intents']:
                last_intent = self.conversation_context['previous_intents'][-1]
                result['negated_intent'] = last_intent

                # Keep track that we've negated the previous intent
                result['intent'] = f"negation_{last_intent}"

                # Extract any alternative service mentioned
                entities = self.extract_entities(text)
                if 'service_type' in entities:
                    result['entities'] = entities
                    result['alternative_requested'] = True
            else:
                # No previous intent to negate, treat as a new request
                result['needs_clarification'] = True
                result['intent'] = 'clarification_needed'

            return result

        # Handle context switching
        elif context_features['contains_context_switch']:
            # Extract entities to understand the new intent
            entities = self.extract_entities(text)
            result['entities'] = entities

            # Try to determine the new intent by extracting entities
            if 'service_type' in entities:
                service_type = entities['service_type'][0].lower()

                # Map service type to likely flow
                if service_type in ['jump start', 'battery', 'tire change', 'lockout']:
                    result['flow'] = 'roadside'
                    result['intent'] = 'request_roadside_specific'
                elif service_type in ['appointment', 'service', 'maintenance', 'oil change']:
                    result['flow'] = 'appointment'
                    result['intent'] = 'book_service_basic'
                else:
                    # If service type doesn't match known types, try intent classification
                    intent = self.predict_intent(text)
                    result['intent'] = intent
            else:
                # No clear service type, use standard intent classification
                intent = self.predict_intent(text)
                result['intent'] = intent

            return result

        # Standard processing for regular messages
        fallback = self.needs_fallback(text)

        if not fallback:
            intent = self.predict_intent(text)
            entities = self.extract_entities(text)

            result['intent'] = intent
            result['entities'] = entities
            result['needs_fallback'] = fallback
        else:
            result['intent'] = 'unknown'
            result['needs_fallback'] = True

        # Update our conversation context
        self.update_conversation_context(result)

        return result

### 4.2 Create Dynamic Clarification Flow

In your `streamlit_app.py` or main application interface, implement a clarification mechanism:

```python
def handle_user_input(user_input, conversation_history):
    """
    Process user input and generate appropriate response with clarification when needed.

    Args:
        user_input: Text from the user
        conversation_history: Previous messages in the conversation

    Returns:
        Response and updated conversation history
    """
    # Process the message
    assistant = load_assistant()
    result = assistant.process_message(user_input)

    # Handle negation and context switching
    if result.get('contains_negation', False):
        if result.get('negated_intent') and result.get('alternative_requested', False):
            # User negated previous request and specified an alternative
            response = generate_alternative_response(result)
        else:
            # Simple negation without alternative
            response = "I understand you don't want that. How else can I help you today?"

    # Handle context switching
    elif result.get('contains_context_switch', False):
        # User switched context to a new request
        response = generate_switch_response(result)

        # Reset any active workflows
        reset_active_workflows()

    # Handle regular clarification needs
    elif result.get('needs_clarification', False):
        # Generate clarification prompt
        response = generate_clarification_prompt(result, conversation_history)

    # Handle fallback case
    elif result.get('needs_fallback', False):
        response = "I'm not sure I understood. Could you please rephrase or provide more details?"

    # Regular flow
    else:
        # Process normally
        response = generate_response(result)

    # Update conversation history
    conversation_history.append({
        'user': user_input,
        'assistant': response,
        'context': result
    })

    return response, conversation_history

def generate_clarification_prompt(result, conversation_history):
    """Generate appropriate clarification based on the current context."""

    # Check what's ambiguous
    if conversation_history and result.get('contains_context_switch', False):
        # Context switch without clear intent
        if 'service_type' not in result.get('entities', {}):
            return "I notice you've changed your request. Could you specify what service you need now?"

    # Check for contradictions
    contradictions = detect_contradictions(result, conversation_history)
    if contradictions:
        return f"I noticed that you mentioned {contradictions[0]} but earlier you mentioned {contradictions[1]}. Could you please clarify what you need?"

    # General clarification
    return "Could you provide more details about what you need assistance with?"

def detect_contradictions(result, conversation_history):
    """
    Detect contradictions between current and previous messages.

    Returns:
        Tuple of (current_element, previous_element) that contradict, or None
    """
    # Only check if we have history
    if not conversation_history:
        return None

    # Get entities from current result
    current_entities = result.get('entities', {})

    # Get entities from last interaction
    last_context = conversation_history[-1].get('context', {})
    last_entities = last_context.get('entities', {})

    # Compare key fields
    contradictions = []

    # Check vehicle details
    for entity_type in ['vehicle_make', 'vehicle_model', 'vehicle_year']:
        if entity_type in current_entities and entity_type in last_entities:
            current_value = current_entities[entity_type][0].lower()
            last_value = last_entities[entity_type][0].lower()

            if current_value != last_value:
                return (f"{entity_type.replace('_', ' ')}: {current_value}",
                        f"{entity_type.replace('_', ' ')}: {last_value}")

    # Check service type
    if 'service_type' in current_entities and 'service_type' in last_entities:
        current_value = current_entities['service_type'][0].lower()
        last_value = last_entities['service_type'][0].lower()

        if current_value != last_value:
            return (f"service: {current_value}", f"service: {last_value}")

    return None
````

## Step 5: Evaluation and Testing

### 5.1 Enhanced Test Case Design

Create dedicated test cases focusing on negation, context switching, and multi-turn interactions:

```python
def create_comprehensive_test_suite():
    """
    Create a comprehensive test suite with specialized test cases for context handling.
    """
    # Basic negation test cases
    negation_test_cases = [
        {
            "test_id": "negation_basic_1",
            "context": {"previous_intent": "request_tow_basic"},
            "input": "I don't need a tow truck.",
            "expected": {
                "contains_negation": True,
                "negated_intent": "request_tow_basic"
            }
        },
        {
            "test_id": "negation_with_alternative_1",
            "context": {"previous_intent": "request_tow_basic"},
            "input": "I don't need a tow, I need a jump start.",
            "expected": {
                "contains_negation": True,
                "negated_intent": "request_tow_basic",
                "alternative_requested": True,
                "entities": {"service_type": ["jump start"]}
            }
        },
        # More negation cases...
    ]

    # Context switching test cases
    context_switch_test_cases = [
        {
            "test_id": "context_switch_1",
            "context": {"previous_intent": "request_tow_basic"},
            "input": "Actually, forget about the tow, can you help me with a flat tire?",
            "expected": {
                "contains_context_switch": True,
                "flow": "roadside",
                "intent": "request_roadside_specific",
                "entities": {"service_type": ["flat tire"]}
            }
        },
        # More context switching cases...
    ]

    # Entity contradiction test cases
    contradiction_test_cases = [
        {
            "test_id": "contradiction_vehicle_1",
            "context": {
                "previous_entities": [
                    {"entity": "vehicle_make", "value": "Toyota"},
                    {"entity": "vehicle_model", "value": "Camry"}
                ]
            },
            "input": "Actually, I'm driving a Honda Civic.",
            "expected": {
                "contains_context_switch": True,
                "entities": {
                    "vehicle_make": ["Honda"],
                    "vehicle_model": ["Civic"]
                },
                "contradicts_previous": True
            }
        },
        # More contradiction cases...
    ]

    # Combine all test cases
    all_test_cases = {
        "negation": negation_test_cases,
        "context_switch": context_switch_test_cases,
        "contradiction": contradiction_test_cases
    }

    return all_test_cases

def create_conversation_simulator():
    """
    Create a simulator for testing multi-turn conversations with realistic user behaviors.
    """
    # Define user simulator behaviors
    behaviors = {
        "standard": {
            "description": "User who follows the expected flow",
            "contradiction_rate": 0.0,
            "negation_rate": 0.1,
            "context_switch_rate": 0.1,
            "ambiguity_rate": 0.1,
        },
        "indecisive": {
            "description": "User who frequently changes their mind",
            "contradiction_rate": 0.2,
            "negation_rate": 0.4,
            "context_switch_rate": 0.3,
            "ambiguity_rate": 0.2,
        },
        "confused": {
            "description": "User who provides contradictory information",
            "contradiction_rate": 0.4,
            "negation_rate": 0.2,
            "context_switch_rate": 0.2,
            "ambiguity_rate": 0.3,
        }
    }

    # Define conversation scenarios
    scenarios = [
        {
            "scenario_id": "tow_then_negation_then_alternative",
            "behavior": "indecisive",
            "initial_intent": "request_tow_basic",
            "turns": 5,
            "success_criteria": {
                "final_intent": "request_roadside_specific",
                "required_entities": ["service_type", "pickup_location"]
            }
        },
        {
            "scenario_id": "contradictory_vehicle_info",
            "behavior": "confused",
            "initial_intent": "vehicle_info_request",
            "turns": 4,
            "success_criteria": {
                "final_entities": ["vehicle_make", "vehicle_model", "vehicle_year"],
                "contradiction_resolved": True
            }
        },
        # More scenarios...
    ]

    return {
        "behaviors": behaviors,
        "scenarios": scenarios
    }

def simulate_conversation(scenario, assistant, simulator_config):
    """
    Run a simulated conversation based on a scenario.

    Args:
        scenario: Conversation scenario definition
        assistant: ContextAwareCarroAssistant instance
        simulator_config: Configuration for user simulator

    Returns:
        Dictionary with conversation log and metrics
    """
    # Initialize conversation
    conversation = []
    behavior = simulator_config["behaviors"][scenario["behavior"]]

    # Get initial message based on intent
    initial_message = get_initial_message_for_intent(scenario["initial_intent"])

    # Process initial message
    result = assistant.process_message(initial_message)
    response = generate_response(result)

    # Store first turn
    conversation.append({
        "turn": 1,
        "user": initial_message,
        "system_result": result,
        "system_response": response
    })

    # Simulation metrics
    metrics = {
        "negations": 0,
        "context_switches": 0,
        "contradictions": 0,
        "ambiguities": 0,
        "successful_recoveries": 0,
        "completion_rate": 0.0
    }

    # Simulate remaining turns
    for turn in range(2, scenario["turns"] + 1):
        # Determine user behavior for this turn
        if random.random() < behavior["negation_rate"]:
            # Generate negation response
            user_message = generate_negation_message(conversation[-1])
            metrics["negations"] += 1
        elif random.random() < behavior["context_switch_rate"]:
            # Generate context switch
            user_message = generate_context_switch_message(conversation[-1])
            metrics["context_switches"] += 1
        elif random.random() < behavior["contradiction_rate"]:
            # Generate contradiction
            user_message = generate_contradiction_message(conversation[-1])
            metrics["contradictions"] += 1
        elif random.random() < behavior["ambiguity_rate"]:
            # Generate ambiguous message
            user_message = generate_ambiguous_message(conversation[-1])
            metrics["ambiguities"] += 1
        else:
            # Generate standard response
            user_message = generate_standard_message(conversation[-1])

        # Process user message
        result = assistant.process_message(user_message)
        response = generate_response(result)

        # Check if the system recovered from a previous issue
        if turn > 2 and (conversation[-2].get("problematic", False) and not conversation[-1].get("problematic", False)):
            metrics["successful_recoveries"] += 1

        # Store this turn
        conversation.append({
            "turn": turn,
            "user": user_message,
            "system_result": result,
            "system_response": response,
            "problematic": result.get("needs_clarification", False) or result.get("needs_fallback", False)
        })

    # Evaluate overall success
    success = evaluate_conversation_success(conversation, scenario["success_criteria"])
    metrics["success"] = success
    metrics["completion_rate"] = calculate_completion_rate(conversation, scenario["success_criteria"])

    return {
        "scenario_id": scenario["scenario_id"],
        "behavior": scenario["behavior"],
        "conversation": conversation,
        "metrics": metrics
    }

def evaluate_context_handling(test_suite, models_dir):
    """
    Evaluate the model's ability to handle negation and context switching.

    Args:
        test_suite: Comprehensive test suite
        models_dir: Directory containing trained models

    Returns:
        Evaluation metrics dictionary with detailed analysis
    """
    # Initialize context-aware assistant
    assistant = ContextAwareCarroAssistant(models_dir)

    # Track metrics for each test category
    metrics = {
        "negation_detection": {
            "correct": 0,
            "total": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        },
        "context_switch_detection": {
            "correct": 0,
            "total": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        },
        "entity_extraction_with_context": {
            "correct": 0,
            "total": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        },
        "contradiction_detection": {
            "correct": 0,
            "total": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    }

    # Process each test category
    for category, test_cases in test_suite.items():
        # Process each test case in this category
        for case in test_cases:
            # Set up conversation context
            if "context" in case:
                assistant.conversation_context = case["context"]
            else:
                # Reset context for cases that don't specify it
                assistant.conversation_context = {
                    "previous_intents": [],
                    "previous_entities": [],
                    "active_flow": None
                }

            # Process the input
            result = assistant.process_message(case["input"])

            # Store detailed case results for analysis
            case_result = {
                "test_id": case["test_id"],
                "input": case["input"],
                "expected": case["expected"],
                "actual": result,
                "success": True  # Will be updated based on evaluation
            }

            # Evaluate based on test category
            if category == "negation":
                evaluate_negation_case(case, result, metrics, case_result)
            elif category == "context_switch":
                evaluate_context_switch_case(case, result, metrics, case_result)
            elif category == "contradiction":
                evaluate_contradiction_case(case, result, metrics, case_result)

            # Record detailed case result
            test_suite[category][test_cases.index(case)]["result"] = case_result

    # Calculate final metrics
    for metric_category in metrics:
        if metrics[metric_category]["total"] > 0:
            metrics[metric_category]["accuracy"] = metrics[metric_category]["correct"] / metrics[metric_category]["total"]

            # Calculate precision, recall, F1 where applicable
            if "true_positives" in metrics[metric_category] and "false_positives" in metrics[metric_category]:
                true_positives = metrics[metric_category]["true_positives"]
                false_positives = metrics[metric_category]["false_positives"]
                false_negatives = metrics[metric_category]["false_negatives"]

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

                metrics[metric_category]["precision"] = precision
                metrics[metric_category]["recall"] = recall
                metrics[metric_category]["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            metrics[metric_category]["accuracy"] = 0

    return {
        "detailed_test_suite": test_suite,
        "summary_metrics": metrics
    }

def run_conversation_simulations(models_dir):
    """
    Run multiple simulated conversations to evaluate system performance.

    Args:
        models_dir: Directory containing trained models

    Returns:
        Summary of simulation results
    """
    # Create simulator configuration
    simulator_config = create_conversation_simulator()

    # Initialize context-aware assistant
    assistant = ContextAwareCarroAssistant(models_dir)

    # Run simulations for each scenario
    scenario_results = []
    for scenario in simulator_config["scenarios"]:
        # Run multiple simulations for statistical significance
        scenario_simulations = []
        for i in range(10):  # Run each scenario 10 times
            simulation_result = simulate_conversation(scenario, assistant, simulator_config)
            scenario_simulations.append(simulation_result)

        # Aggregate metrics across simulations
        aggregated_metrics = aggregate_simulation_metrics(scenario_simulations)

        # Record results
        scenario_results.append({
            "scenario_id": scenario["scenario_id"],
            "behavior": scenario["behavior"],
            "metrics": aggregated_metrics,
            "success_rate": sum(sim["metrics"]["success"] for sim in scenario_simulations) / len(scenario_simulations)
        })

    # Calculate conversation-level metrics
    conversation_metrics = {
        "overall_success_rate": sum(scenario["success_rate"] for scenario in scenario_results) / len(scenario_results),
        "negation_recovery_rate": sum(scenario["metrics"]["successful_recoveries"] / max(1, scenario["metrics"]["negations"])
                                     for scenario in scenario_results) / len(scenario_results),
        "context_switch_recovery_rate": sum(scenario["metrics"]["successful_recoveries"] / max(1, scenario["metrics"]["context_switches"])
                                          for scenario in scenario_results) / len(scenario_results),
        "contradiction_recovery_rate": sum(scenario["metrics"]["successful_recoveries"] / max(1, scenario["metrics"]["contradictions"])
                                         for scenario in scenario_results) / len(scenario_results)
    }

    return {
        "scenario_results": scenario_results,
        "conversation_metrics": conversation_metrics
    }
```

### 5.2 Comprehensive Evaluation Framework

Update the evaluation script (`evaluation.py`) to include specific metrics for negation and context switching with detailed analysis:

```python
def evaluate_negation_context_handling(test_data_dir: str, models_dir: str) -> Dict[str, Any]:
    """
    Evaluate the model's handling of negation and context switching with comprehensive metrics.

    Args:
        test_data_dir: Directory containing test data
        models_dir: Directory containing trained models

    Returns:
        Dictionary of evaluation metrics with detailed analysis
    """
    # Load test suite
    test_suite = create_comprehensive_test_suite()

    # Run single-turn evaluations
    single_turn_results = evaluate_context_handling(test_suite, models_dir)

    # Run multi-turn conversation simulations
    conversation_results = run_conversation_simulations(models_dir)

    # Create analysis visualizations
    create_evaluation_visualizations(single_turn_results, conversation_results, models_dir)

    # Combine results with deeper analysis
    combined_results = {
        "single_turn_metrics": single_turn_results["summary_metrics"],
        "conversation_metrics": conversation_results["conversation_metrics"],
        "scenario_results": conversation_results["scenario_results"],
        "detailed_analysis": {
            "negation_patterns": analyze_negation_patterns(single_turn_results["detailed_test_suite"]["negation"]),
            "context_switch_patterns": analyze_context_switch_patterns(single_turn_results["detailed_test_suite"]["context_switch"]),
            "contradiction_patterns": analyze_contradiction_patterns(single_turn_results["detailed_test_suite"]["contradiction"]),
            "error_analysis": perform_error_analysis(single_turn_results, conversation_results)
        }
    }

    # Save results to file
    save_path = os.path.join(models_dir, "evaluation", "context_handling_evaluation.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Generate human-readable report
    generate_context_handling_report(combined_results, models_dir)

    return combined_results
```

## Implementation Steps

To implement these changes, follow these steps:

1. Create the negation examples dataset (`negation_examples.json`)
2. Update the data augmentation scripts with negation and context switching functions
3. Implement the contextual intent classifier and multi-task clarification model
4. Update the inference process to incorporate context awareness
5. Create test cases for negation and context switching
6. Evaluate the enhanced system

### Implementation Timeline:

1. **Day 1**: Data preparation and augmentation

   - Create negation examples
   - Update augmentation scripts
   - Run data augmentation pipeline

2. **Day 2-3**: Model architecture updates

   - Implement contextual intent classifier
   - Implement multi-task learning for clarification detection
   - Train models on enhanced datasets

3. **Day 4**: Inference engine updates

   - Implement context-aware inference process
   - Create clarification mechanism
   - Integrate with existing code

4. **Day 5**: Testing and evaluation
   - Create specialized test cases
   - Run comprehensive evaluations
   - Analyze results and make adjustments

### Success Metrics:

Measure the following metrics to evaluate the effectiveness of the improvements:

1. **Negation Detection Accuracy**: Ability to correctly identify when a user negates a previous request
2. **Context Switch Detection Accuracy**: Ability to identify when a user changes the topic/intent
3. **Entity Extraction with Context**: Accuracy of entity extraction in context-sensitive situations
4. **Multi-turn Success Rate**: Success rate of handling realistic multi-turn conversations
5. **Overall User Satisfaction**: Test with real users and measure satisfaction scores

## Conclusion

By implementing these enhancements, your chatbot will significantly improve its ability to handle negation and context switching. The multi-task learning approach will enable the system to simultaneously detect negation, context switches, and clarification needs, while the context-aware inference process will ensure appropriate responses based on conversation history.

This implementation follows ChatGPT's recommendations but adds concrete code examples and a detailed implementation plan specific to your existing codebase.
