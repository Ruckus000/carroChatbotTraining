# Implementation Plan for Improved Context Handling

## Problem Overview

Your chatbot is currently built on a DistilBERT-based architecture for intent classification and entity extraction, but it struggles with negation (e.g., "I don't need a tow truck") and context switching (e.g., "Actually, forget the tow truck; I need a new battery"). This implementation plan will enhance your system to better handle these scenarios.

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

Modify your model_training.py to include context sensitivity in the intent classifier:

```python
class ContextualIntentClassifier(DistilBertForSequenceClassification):
    """DistilBERT model with context awareness for intent classification"""

    def __init__(self, config):
        super().__init__(config)
        # Add a component to handle negation features
        self.negation_detector = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # Get standard outputs from DistilBERT
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Extract last hidden state for negation detection
        sequence_output = outputs.hidden_states[-1]
        pooled_output = sequence_output[:, 0]  # Use CLS token

        # Detect negation
        negation_logits = self.negation_detector(pooled_output)

        # Include negation prediction in outputs
        outputs.negation_logits = negation_logits

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

    # Add negation and context switching flags
    train_negation = [1 if "don't" in x['text'].lower() or "not " in x['text'].lower() or
                     "no longer" in x['text'].lower() or "forget" in x['text'].lower() else 0
                     for x in flow_train_data]

    val_negation = [1 if "don't" in x['text'].lower() or "not " in x['text'].lower() or
                    "no longer" in x['text'].lower() or "forget" in x['text'].lower() else 0
                    for x in flow_val_data]

    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Add negation flag to encodings (this would need to be handled in a custom dataset)
    train_encodings['negation'] = train_negation
    val_encodings['negation'] = val_negation

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

Update your training approach to incorporate multi-task learning:

```python
def train_multi_task_clarification_classifier(dataset_dir: str, output_dir: str):
    """
    Train a multi-task classifier that detects both clarification needs and negation.

    Args:
        dataset_dir: Directory containing datasets
        output_dir: Directory to save model
    """
    logger.info("Training multi-task clarification classifier")

    # Load datasets
    train_file = os.path.join(dataset_dir, 'clarification_classification_train.json')
    val_file = os.path.join(dataset_dir, 'clarification_classification_val.json')

    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(val_file, 'r') as f:
        val_data = json.load(f)

    # Prepare multi-task labels
    train_texts = [item['text'] for item in train_data]
    train_clarification_labels = [1 if item['label'] == 'clarification' else 0 for item in train_data]
    train_negation_labels = [1 if item.get('type') == 'negation' else 0 for item in train_data]
    train_context_switch_labels = [1 if item.get('type') == 'context_switch' else 0 for item in train_data]

    val_texts = [item['text'] for item in val_data]
    val_clarification_labels = [1 if item['label'] == 'clarification' else 0 for item in val_data]
    val_negation_labels = [1 if item.get('type') == 'negation' else 0 for item in val_data]
    val_context_switch_labels = [1 if item.get('type') == 'context_switch' else 0 for item in val_data]

    # Create a custom dataset for multi-task learning
    class MultiTaskDataset(Dataset):
        def __init__(self, encodings, clarification_labels, negation_labels, context_switch_labels):
            self.encodings = encodings
            self.clarification_labels = clarification_labels
            self.negation_labels = negation_labels
            self.context_switch_labels = context_switch_labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['clarification_labels'] = torch.tensor(self.clarification_labels[idx])
            item['negation_labels'] = torch.tensor(self.negation_labels[idx])
            item['context_switch_labels'] = torch.tensor(self.context_switch_labels[idx])
            return item

        def __len__(self):
            return len(self.clarification_labels)

    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Create datasets
    train_dataset = MultiTaskDataset(train_encodings, train_clarification_labels, train_negation_labels, train_context_switch_labels)
    val_dataset = MultiTaskDataset(val_encodings, val_clarification_labels, val_negation_labels, val_context_switch_labels)

    # Create a custom multi-task model
    class MultiTaskClarificationModel(DistilBertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.distilbert = DistilBertModel(config)
            self.pre_classifier = torch.nn.Linear(config.dim, config.dim)
            self.dropout = torch.nn.Dropout(config.seq_classif_dropout)

            # Multiple classification heads
            self.clarification_classifier = torch.nn.Linear(config.dim, 2)
            self.negation_classifier = torch.nn.Linear(config.dim, 2)
            self.context_switch_classifier = torch.nn.Linear(config.dim, 2)

            self.init_weights()

        def forward(self, input_ids=None, attention_mask=None,
                  clarification_labels=None, negation_labels=None, context_switch_labels=None):
            # DistilBERT forward pass
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get sequence output and apply pre-classifier and dropout
            hidden_state = outputs[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)

            # Multiple classification heads
            clarification_logits = self.clarification_classifier(pooled_output)  # (bs, 2)
            negation_logits = self.negation_classifier(pooled_output)  # (bs, 2)
            context_switch_logits = self.context_switch_classifier(pooled_output)  # (bs, 2)

            loss = None
            if clarification_labels is not None and negation_labels is not None and context_switch_labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                clarification_loss = loss_fct(clarification_logits.view(-1, 2), clarification_labels.view(-1))
                negation_loss = loss_fct(negation_logits.view(-1, 2), negation_labels.view(-1))
                context_switch_loss = loss_fct(context_switch_logits.view(-1, 2), context_switch_labels.view(-1))

                # Combined loss with weights (can be adjusted)
                loss = clarification_loss + 0.5 * negation_loss + 0.5 * context_switch_loss

            return {
                'loss': loss,
                'clarification_logits': clarification_logits,
                'negation_logits': negation_logits,
                'context_switch_logits': context_switch_logits
            }

    # Initialize model
    model = MultiTaskClarificationModel.from_pretrained('distilbert-base-uncased')

    # Rest of the training code...
    # ...
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

        # Maintain conversation context
        self.conversation_context = {
            "previous_intents": [],
            "previous_entities": [],
            "active_flow": None
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

        Args:
            processing_result: Result from process_message
        """
        # Store previous intent in context
        if processing_result['intent'] not in ['unknown', 'clarification_needed']:
            self.conversation_context['previous_intents'].append(processing_result['intent'])
            if len(self.conversation_context['previous_intents']) > 3:
                self.conversation_context['previous_intents'].pop(0)  # Keep last 3 intents

        # Store entities
        for entity_type, values in processing_result['entities'].items():
            for value in values:
                self.conversation_context['previous_entities'].append({
                    'entity': entity_type,
                    'value': value
                })

        # Update active flow
        if processing_result.get('flow') and processing_result['flow'] not in ['clarification', 'fallback']:
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

### 5.1 Create Specialized Test Cases

Create dedicated test cases focusing on negation and context switching:

```python
def create_negation_test_cases():
    """
    Create test cases specifically for evaluating negation handling.
    """
    test_cases = [
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
        {
            "test_id": "complex_negation_1",
            "context": {"previous_intent": "vehicle_info_request"},
            "input": "It's not a Toyota, it's a Honda Civic.",
            "expected": {
                "contains_negation": True,
                "entities": {
                    "vehicle_make": ["Honda"],
                    "vehicle_model": ["Civic"]
                }
            }
        }
    ]

    return test_cases

def evaluate_context_handling(test_cases, models_dir):
    """
    Evaluate the model's ability to handle negation and context switching.

    Args:
        test_cases: List of test case dictionaries
        models_dir: Directory containing trained models

    Returns:
        Evaluation metrics dictionary
    """
    # Initialize context-aware assistant
    assistant = ContextAwareCarroAssistant(models_dir)

    # Track metrics
    metrics = {
        "negation_detection": {
            "correct": 0,
            "total": 0
        },
        "context_switch_detection": {
            "correct": 0,
            "total": 0
        },
        "entity_extraction_with_context": {
            "correct": 0,
            "total": 0
        }
    }

    # Process each test case
    for case in test_cases:
        # Set up conversation context
        if "context" in case:
            assistant.conversation_context = case["context"]

        # Process the input
        result = assistant.process_message(case["input"])

        # Compare with expected output
        expected = case["expected"]

        # Check negation detection
        if "contains_negation" in expected:
            metrics["negation_detection"]["total"] += 1
            if result.get("contains_negation") == expected["contains_negation"]:
                metrics["negation_detection"]["correct"] += 1

        # Check context switch detection
        if "contains_context_switch" in expected:
            metrics["context_switch_detection"]["total"] += 1
            if result.get("contains_context_switch") == expected["contains_context_switch"]:
                metrics["context_switch_detection"]["correct"] += 1

        # Check entity extraction
        if "entities" in expected:
            metrics["entity_extraction_with_context"]["total"] += 1
            entities_match = True

            for entity_type, expected_values in expected["entities"].items():
                if entity_type not in result.get("entities", {}) or set(expected_values) != set(result["entities"][entity_type]):
                    entities_match = False
                    break

            if entities_match:
                metrics["entity_extraction_with_context"]["correct"] += 1

    # Calculate accuracy for each metric
    for metric in metrics:
        if metrics[metric]["total"] > 0:
            metrics[metric]["accuracy"] = metrics[metric]["correct"] / metrics[metric]["total"]
        else:
            metrics[metric]["accuracy"] = 0

    return metrics

def run_multi_turn_conversation_test(models_dir):
    """
    Test the system with multi-turn conversations to evaluate context handling.

    Args:
        models_dir: Directory containing trained models

    Returns:
        Evaluation results
    """
    # Define multi-turn test scenarios
    scenarios = [
        {
            "scenario_id": "tow_then_negation_then_alternative",
            "turns": [
                {
                    "user": "I need a tow truck",
                    "expected_flow": "towing",
                    "expected_intent": "request_tow_basic"
                },
                {
                    "user": "Actually I don't need a tow truck",
                    "expected_contains_negation": True,
                    "expected_negated_intent": "request_tow_basic"
                },
                {
                    "user": "I need help with a flat tire",
                    "expected_flow": "roadside",
                    "expected_intent": "request_roadside_specific",
                    "expected_entities": {"service_type": ["flat tire"]}
                }
            ]
        },
        {
            "scenario_id": "roadside_then_context_switch",
            "turns": [
                {
                    "user": "My car battery is dead",
                    "expected_flow": "roadside",
                    "expected_intent": "request_roadside_specific"
                },
                {
                    "user": "Actually, can you help me schedule an oil change next week?",
                    "expected_contains_context_switch": True,
                    "expected_flow": "appointment",
                    "expected_intent": "book_service_basic"
                }
            ]
        }
    ]

    # Initialize context-aware assistant
    assistant = ContextAwareCarroAssistant(models_dir)

    # Track results
    results = []

    # Run each scenario
    for scenario in scenarios:
        scenario_result = {
            "scenario_id": scenario["scenario_id"],
            "turns": [],
            "success": True
        }

        # Reset assistant context
        assistant.conversation_context = {
            "previous_intents": [],
            "previous_entities": [],
            "active_flow": None
        }

        # Process each turn
        for turn in scenario["turns"]:
            # Get user input
            user_input = turn["user"]

            # Process message
            result = assistant.process_message(user_input)

            # Check expectations
            turn_result = {
                "user": user_input,
                "system_result": result,
                "expectations_met": True,
                "failures": []
            }

            # Check flow
            if "expected_flow" in turn and result.get("flow") != turn["expected_flow"]:
                turn_result["expectations_met"] = False
                turn_result["failures"].append(f"Flow mismatch: expected {turn['expected_flow']}, got {result.get('flow')}")

            # Check intent
            if "expected_intent" in turn and result.get("intent") != turn["expected_intent"]:
                turn_result["expectations_met"] = False
                turn_result["failures"].append(f"Intent mismatch: expected {turn['expected_intent']}, got {result.get('intent')}")

            # Check negation
            if "expected_contains_negation" in turn and result.get("contains_negation") != turn["expected_contains_negation"]:
                turn_result["expectations_met"] = False
                turn_result["failures"].append(f"Negation detection mismatch: expected {turn['expected_contains_negation']}, got {result.get('contains_negation')}")

            # Check context switch
            if "expected_contains_context_switch" in turn and result.get("contains_context_switch") != turn["expected_contains_context_switch"]:
                turn_result["expectations_met"] = False
                turn_result["failures"].append(f"Context switch detection mismatch: expected {turn['expected_contains_context_switch']}, got {result.get('contains_context_switch')}")

            # Check entities
            if "expected_entities" in turn:
                for entity_type, expected_values in turn["expected_entities"].items():
                    if entity_type not in result.get("entities", {}) or set(expected_values) != set(result["entities"][entity_type]):
                        turn_result["expectations_met"] = False
                        turn_result["failures"].append(f"Entity mismatch for {entity_type}: expected {expected_values}, got {result.get('entities', {}).get(entity_type, [])}")

            # Add turn result to scenario
            scenario_result["turns"].append(turn_result)

            # Update scenario success
            if not turn_result["expectations_met"]:
                scenario_result["success"] = False

        # Add scenario result
        results.append(scenario_result)

    # Calculate overall success rate
    success_count = sum(1 for scenario in results if scenario["success"])
    overall_success_rate = success_count / len(results) if results else 0

    return {
        "scenario_results": results,
        "overall_success_rate": overall_success_rate
    }
```

### 5.2 Evaluate on Specific Negation and Context Metrics

Update the evaluation script (`evaluation.py`) to include specific metrics for negation and context switching:

```python
def evaluate_negation_context_handling(test_data_dir: str, models_dir: str) -> Dict[str, Any]:
    """
    Evaluate the model's handling of negation and context switching.

    Args:
        test_data_dir: Directory containing test data
        models_dir: Directory containing trained models

    Returns:
        Dictionary of evaluation metrics
    """
    # Load negation test data
    with open(os.path.join(test_data_dir, "negation_context_test.json"), 'r') as f:
        test_cases = json.load(f)

    # Run evaluation
    metrics = evaluate_context_handling(test_cases, models_dir)

    # Run multi-turn tests
    conversation_results = run_multi_turn_conversation_test(models_dir)

    # Combine results
    combined_results = {
        "single_turn_metrics": metrics,
        "multi_turn_results": conversation_results
    }

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
