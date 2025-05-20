# Fine-tuning Plan for Domain-Specific Sentiment Model

## Overview

This document outlines a plan to fine-tune a sentiment analysis model specifically for the automotive assistance domain. Currently, the chatbot uses a general-purpose DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) which was trained on movie reviews. A domain-specific model would better capture sentiment nuances in automotive assistance scenarios, improving urgent situation detection and response appropriateness.

## 1. Data Collection and Preparation with ChatGPT

### 1.1 ChatGPT-Based Data Generation Approach

- **Primary Data Source**: Use ChatGPT to generate large volumes of domain-specific, sentiment-labeled data
- **Technical Approach**:
  - Direct prompting of ChatGPT (GPT-3.5-Turbo or GPT-4)
  - Batch generation with structured output formats
  - Automated collection via API if volume requires
- **Advantages**:
  - No need for manual annotation
  - Rapid generation of domain-specific examples
  - Built-in automotive domain knowledge
  - Consistent labeling across sentiment categories

### 1.2 Sentiment Category Definition

- **Four-Category System**:
  - **Urgent Negative**: High priority issues requiring immediate attention (e.g., "My car broke down on the highway and I have children with me")
  - **Standard Negative**: Regular complaints or problems (e.g., "The air conditioning isn't working properly")
  - **Neutral**: Informational or general inquiries (e.g., "What are your service hours?")
  - **Positive**: Compliments or satisfaction expressions (e.g., "The service was excellent")

### 1.3 ChatGPT Prompt Engineering

- **Prompt Structure**:
  - Clear task description with desired output format
  - Detailed category definitions with examples
  - Scenario parameters to ensure diversity
  - Output format specification with JSON structure
  - System instructions to maintain consistent tone and realism

- **Example Base Prompt**:
  ```
  Generate realistic automotive assistance conversations with sentiment labels.
  For each example, include:
  1. A customer message
  2. The sentiment category (urgent_negative, standard_negative, neutral, positive)
  3. A brief justification for the chosen category

  Output in JSON format:
  {"text": "customer message", "sentiment": "sentiment_category"}
  ```

- **Diversity Strategy**:
  - Vary vehicle types, issues, customer demographics
  - Include different regional language patterns
  - Mix formal and informal communication styles
  - Balance technical and non-technical vocabulary

### 1.4 Data Generation Workflow with ChatGPT

1. **Initial Data Generation**:
   - Use the comprehensive prompt template provided in section 1.6
   - Generate 50-100 examples per batch
   - Request equal distribution across sentiment categories
   - Run 8-10 batches with slightly varied prompts

2. **Data Augmentation**:
   - Request paraphrases of generated examples
   - Introduce controlled variations (time of day, weather conditions, severity)
   - Generate "conversation chains" showing sentiment progression
   - Create edge cases for testing boundary conditions

3. **Quality Control**:
   - Manual review of ~10% of generated data
   - Check for category accuracy and realism
   - Ensure balanced distribution across categories
   - Verify sentence structure and language naturalism

4. **Dataset Compilation**:
   - Convert all outputs to consistent JSON format
   - De-duplicate based on semantic similarity
   - Balance categories if necessary
   - Split into train/validation/test (70%/15%/15%)

### 1.5 Data Processing Pipeline

```python
# Python script outline for ChatGPT data processing
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw ChatGPT outputs
def load_chatgpt_outputs(input_dir):
    all_examples = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r') as f:
                batch_data = json.load(f)
                all_examples.extend(batch_data)
    return all_examples

# Process and clean the data
def process_examples(examples):
    processed = []
    for ex in examples:
        # Basic cleaning
        text = ex.get('text', '').strip()
        sentiment = ex.get('sentiment', '').strip()
        
        # Skip invalid examples
        if not text or not sentiment:
            continue
            
        # Normalize sentiment labels
        if sentiment.lower() in ['urgent negative', 'urgent_negative', 'urgent-negative']:
            sentiment = 'urgent_negative'
        elif sentiment.lower() in ['standard negative', 'standard_negative', 'negative']:
            sentiment = 'standard_negative'
        elif sentiment.lower() in ['neutral']:
            sentiment = 'neutral'
        elif sentiment.lower() in ['positive']:
            sentiment = 'positive'
        else:
            continue  # Skip examples with invalid sentiment
            
        processed.append({
            'text': text,
            'sentiment': sentiment
        })
    return processed

# Main processing function
def prepare_sentiment_dataset(input_dir, output_file, test_size=0.15, val_size=0.15):
    # Load and process data
    raw_examples = load_chatgpt_outputs(input_dir)
    processed_examples = process_examples(raw_examples)
    
    # Ensure balanced distribution
    df = pd.DataFrame(processed_examples)
    print(f"Category distribution:\n{df['sentiment'].value_counts()}")
    
    # Split the data
    train_data, test_data = train_test_split(
        processed_examples, test_size=test_size, 
        stratify=[ex['sentiment'] for ex in processed_examples]
    )
    
    train_data, val_data = train_test_split(
        train_data, test_size=val_size/(1-test_size), 
        stratify=[ex['sentiment'] for ex in train_data]
    )
    
    # Save the datasets
    dataset = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset with {len(train_data)} training examples, "
          f"{len(val_data)} validation examples, and {len(test_data)} test examples")
    return dataset
```

### 1.6 Complete ChatGPT Prompt Template

```
# Automotive Sentiment Training Data Generation

## Task
Generate realistic customer messages for an automotive assistance chatbot that reflect different sentiment categories. Focus on creating diverse, authentic-sounding messages that would appear in real automotive assistance conversations.

## Requirements
- Create [NUMBER] examples for each sentiment category (urgent_negative, standard_negative, neutral, positive)
- Vary the automotive problems, contexts, and communication styles
- Ensure natural language patterns and realistic automotive terminology
- Include variations in message length, complexity, and formality

## Sentiment Categories and Examples

1. **urgent_negative** - Messages expressing high urgency, distress, or anger about critical automotive issues requiring immediate assistance
   - "My car suddenly died on the interstate and I'm stranded with my kids in the rain. Need help ASAP!"
   - "There's smoke pouring out from under my hood and I don't know what to do!"
   - "I've been waiting for roadside assistance for THREE HOURS in freezing weather!"

2. **standard_negative** - Messages expressing disappointment, frustration, or annoyance about non-critical automotive issues
   - "My check engine light keeps coming on despite three visits to your shop."
   - "The quote you gave me was way off from the final bill. Not happy about this."
   - "I don't think your technician fixed the problem correctly. The noise is still there."

3. **neutral** - Factual, information-seeking, or routine messages without strong emotional content
   - "When do you close today? I need to drop my car off for service."
   - "Do you have the replacement wiper blades for a 2020 Honda Accord?"
   - "Can you tell me how much an oil change typically costs?"

4. **positive** - Messages expressing satisfaction, gratitude, or positive experiences
   - "Your mechanic did an amazing job with my brake repair! Car stops perfectly now."
   - "Thanks for fitting me in last minute yesterday. Really appreciate the help!"
   - "I've been coming to your shop for 5 years and always get great service."

## Automotive Contexts to Include

- Breakdowns and emergencies
- Regular maintenance
- Repairs and part replacements  
- Service scheduling
- Cost and billing issues
- Vehicle performance problems
- Routine inquiries
- Follow-up communications
- Technical questions
- Warranty and coverage questions

## Output Format
Please provide each example in JSON format:

```json
{"text": "customer message here", "sentiment": "sentiment_category_here"}
```

Where sentiment_category_here is one of: urgent_negative, standard_negative, neutral, positive

Generate [NUMBER] examples per category for a total of [TOTAL] examples.
```

## 2. Model Selection

### 2.1 Base Model Options and Selection

**_Selected Model: RoBERTa_**

After evaluating different options, RoBERTa is selected as the optimal model for the automotive assistance domain due to:

- Superior performance on sentiment classification tasks
- Better ability to detect nuanced emotions (crucial for distinguishing urgent vs. standard negative sentiment)
- Strong contextual understanding for automotive assistance scenarios
- Extensive fine-tuning examples specifically for sentiment tasks

Other considered options:

- **DistilBERT**: Lightweight, fast inference (currently used)
- **BERT-base**: More parameters than DistilBERT
- **ALBERT**: Parameter-efficient version with similar performance to BERT

### 2.2 Considerations for MacBook M4 with 36GB RAM

- **Hardware Advantages**:
  - M4 chip provides excellent ML acceleration via MPS (Metal Performance Shaders)
  - 36GB RAM enables training larger models with bigger batch sizes
  - Apple Silicon optimizations available in recent PyTorch versions

- **Model Size Options**:
  - RoBERTa-large (355M parameters) is fully viable with this hardware
  - Sequence length can be increased to 256 or 512 with minimal performance impact
  - Multiple models can be kept in memory simultaneously for ensemble methods

- **Performance Expectations**:
  - Training speed: ~3-4x faster than CPU-only training
  - Inference latency: <30ms per request on optimized models
  - Can handle concurrent model loading for intent, entity, and sentiment classifiers

## 3. Fine-tuning Approach

### 3.1 Training Strategy for M4 MacBook

- **Transfer Learning**: Start with pre-trained weights and fine-tune on domain data
- **Learning Rate**: Use a small learning rate (1e-5 to 5e-5) with linear decay
- **Epochs**: Start with 3-5 epochs and use early stopping based on validation loss
- **M4-Optimized Settings**:
  - **Batch Size**: 24-32 (much larger than typical due to 36GB RAM)
  - **Mixed Precision**: Use `float16` precision with MPS backend for 2-3x speed boost
  - **Sequence Length**: Safely use 256 tokens (increased from 128)
  - **Model Options**: RoBERTa-large is fully viable, or use RoBERTa-base with larger batches

### 3.2 Technical Setup

- **Framework**: Hugging Face Transformers for model fine-tuning
- **PyTorch Configuration**:
  ```python
  # Optimal PyTorch configuration for M4 MacBook
  import torch
  if torch.backends.mps.is_available() and torch.backends.mps.is_built():
      device = torch.device("mps")
      print("Using MPS (Apple Silicon) acceleration")
  else:
      device = torch.device("cpu")
  
  # Enable mixed precision for M4
  mixed_precision_dtype = torch.float16  # Much faster on M4
  ```

- **Model Loading**:
  ```python
  # Use HuggingFace accelerate library for additional speedups
  from accelerate import infer_auto_device_map, init_empty_weights
  
  # Can use RoBERTa-large with 36GB RAM
  model_name = "roberta-large"  # 355M parameters
  ```

### 3.3 Hyperparameter Tuning

- **Parameters to Optimize**:
  - Learning rate
  - Batch size
  - Dropout rate
  - Weight decay
- **M4-Optimized Starting Parameters**:
  ```
  learning_rate: 2e-5
  batch_size: 32  # Higher due to 36GB RAM
  weight_decay: 0.01
  warmup_steps: 200
  gradient_accumulation_steps: 1  # Not needed with larger batches
  fp16: True  # Enable mixed precision
  ```

### 3.4 Training Code Implementation

```python
def train_sentiment_classifier(data, test_size=0.2, batch_size=32, num_epochs=5):
    """Train the sentiment classifier model using RoBERTa on Apple Silicon."""
    # Make sure the model output directory exists
    sentiment_model_dir = model_file_path("sentiment_model")
    ensure_dir_exists(sentiment_model_dir)

    # Process sentiment data
    X = [example["text"] for example in data if "sentiment" in example]
    sentiments = list(set(example["sentiment"] for example in data if "sentiment" in example))
    sentiments.sort()  # Sort for deterministic result
    
    # Create mapping dictionaries
    sentiment2id = {sentiment: i for i, sentiment in enumerate(sentiments)}
    id2sentiment = {i: sentiment for sentiment, i in sentiment2id.items()}
    
    # Save mapping
    with open(model_file_path("sentiment_model/sentiment2id.json"), "w") as f:
        json.dump(sentiment2id, f, indent=2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, [sentiment2id[example["sentiment"]] for example in data if "sentiment" in example],
        test_size=test_size, random_state=42, stratify=[sentiment2id[example["sentiment"]] 
                                                      for example in data if "sentiment" in example]
    )
    
    # Load tokenizer
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    
    # Can use RoBERTa-large with 36GB RAM and M4 chip
    model_name = "roberta-large"
    sentiment_tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Tokenize with larger sequence length (256 instead of 128) - viable on M4
    train_sentiment_encodings = sentiment_tokenizer(
        X_train, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    val_sentiment_encodings = sentiment_tokenizer(
        X_test, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    
    # Create datasets
    train_sentiment_dataset = Dataset.from_dict({
        "input_ids": train_sentiment_encodings["input_ids"],
        "attention_mask": train_sentiment_encodings["attention_mask"],
        "labels": y_train,
    })
    
    val_sentiment_dataset = Dataset.from_dict({
        "input_ids": val_sentiment_encodings["input_ids"],
        "attention_mask": val_sentiment_encodings["attention_mask"],
        "labels": y_test,
    })
    
    # Load model - we can use larger model with M4 + 36GB RAM
    sentiment_model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(sentiment2id),
        id2label=id2sentiment,
        label2id=sentiment2id,
    )
    sentiment_model.to(DEVICE)
    
    # Configure training with M4 optimizations
    sentiment_training_args = TrainingArguments(
        output_dir=model_file_path("sentiment_model_checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,  # Use ratio instead of steps
        weight_decay=0.01,
        logging_dir=model_file_path("sentiment_logs"),
        logging_steps=20,  # More frequent logging with faster training
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Enable mixed precision for M4
        fp16=True,  # Much faster on Apple Silicon
        fp16_full_eval=True,
        # Early stopping
        early_stopping_patience=2,
    )
    
    # Create and run trainer
    sentiment_trainer = Trainer(
        model=sentiment_model,
        args=sentiment_training_args,
        train_dataset=train_sentiment_dataset,
        eval_dataset=val_sentiment_dataset,
        compute_metrics=compute_sentiment_metrics,
    )
    
    # Train model
    print("Training sentiment model with M4 optimizations...")
    sentiment_trainer.train()
    
    # Save model
    sentiment_model.save_pretrained(model_file_path("sentiment_model"))
    sentiment_tokenizer.save_pretrained(model_file_path("sentiment_model"))
    
    print("Sentiment classifier training complete!")
```

## A. ChatGPT Data Generation Script

To automate the process of generating data with ChatGPT, you can use this Python script that leverages the ChatGPT API:

```python
import os
import json
import time
import argparse
import openai

# Configure OpenAI API (requires API key set as environment variable OPENAI_API_KEY)
openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_sentiment_data(
    num_examples_per_category=50,
    model="gpt-3.5-turbo",
    output_file="chatgpt_sentiment_data.json",
    temperature=0.7
):
    """
    Generate automotive sentiment data using ChatGPT API
    
    Args:
        num_examples_per_category: Number of examples per sentiment category
        model: ChatGPT model to use (gpt-3.5-turbo or gpt-4)
        output_file: File to save the generated data
        temperature: Creative temperature (0.0-1.0)
    """
    # Calculate total examples
    total_examples = num_examples_per_category * 4  # 4 sentiment categories
    
    # Create prompt using template
    with open("prompt_template.txt", "r") as f:
        prompt_template = f.read()
    
    # Replace placeholders
    prompt = prompt_template.replace("[NUMBER]", str(num_examples_per_category))
    prompt = prompt.replace("[TOTAL]", str(total_examples))
    
    # Call ChatGPT API
    print(f"Generating {total_examples} examples ({num_examples_per_category} per category)...")
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic automotive customer service data."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4000,
            n=1,
            stop=None,
        )
        
        # Extract response text
        result = response.choices[0].message.content
        
        # Parse the JSON responses
        # This handles the case where ChatGPT returns JSON lines (one per line)
        examples = []
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('{"text":'):
                try:
                    example = json.loads(line.rstrip(','))
                    examples.append(example)
                except json.JSONDecodeError:
                    pass
                    
        # If no examples parsed, try to extract from markdown code blocks
        if not examples:
            import re
            json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
            for block in json_blocks:
                try:
                    # Try parsing as array
                    parsed_data = json.loads(block)
                    if isinstance(parsed_data, list):
                        examples.extend(parsed_data)
                    else:
                        examples.append(parsed_data)
                except json.JSONDecodeError:
                    # Try parsing line by line
                    for line in block.split('\n'):
                        line = line.strip()
                        if line.startswith('{"text":'):
                            try:
                                example = json.loads(line.rstrip(','))
                                examples.append(example)
                            except json.JSONDecodeError:
                                pass
        
        # Save the examples
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
            
        # Calculate statistics
        categories = {}
        for ex in examples:
            sentiment = ex.get('sentiment', 'unknown')
            categories[sentiment] = categories.get(sentiment, 0) + 1
            
        print(f"Generated {len(examples)} examples")
        print("Sentiment distribution:")
        for category, count in categories.items():
            print(f"  {category}: {count}")
        
        return examples
        
    except Exception as e:
        print(f"Error generating data: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate automotive sentiment data with ChatGPT")
    parser.add_argument("--examples", type=int, default=50, 
                        help="Number of examples per sentiment category")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                        choices=["gpt-3.5-turbo", "gpt-4"], 
                        help="ChatGPT model to use")
    parser.add_argument("--output", type=str, default="data/chatgpt_sentiment_data.json", 
                        help="Output file path")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation (0.0-1.0)")
    parser.add_argument("--batches", type=int, default=1, 
                        help="Number of batches to generate")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data in batches
    all_examples = []
    
    for i in range(args.batches):
        batch_output = f"{os.path.splitext(args.output)[0]}_batch{i+1}.json"
        print(f"\nGenerating batch {i+1}/{args.batches}...")
        
        examples = generate_sentiment_data(
            num_examples_per_category=args.examples,
            model=args.model,
            output_file=batch_output,
            temperature=args.temperature
        )
        
        all_examples.extend(examples)
        
        # Wait a bit between batches to avoid API rate limits
        if i < args.batches - 1:
            print("Waiting before next batch...")
            time.sleep(5)
    
    # Combine all batches
    if args.batches > 1:
        with open(args.output, 'w') as f:
            json.dump(all_examples, f, indent=2)
        print(f"\nCombined {len(all_examples)} examples into {args.output}")
```

## 4. Evaluation

### 4.1 Metrics

- **Primary Metrics**:
  - F1 score (macro-averaged across classes)
  - Accuracy
  - Class-specific precision/recall (especially for urgent negative)
- **Secondary Metrics**:
  - Confusion matrix analysis
  - ROC/AUC curves

### 4.2 Benchmarking

- **Baseline**: Compare RoBERTa performance against current DistilBERT model
- **Integration with Existing Dashboard**: Use NLU dashboard for visualization
- **Regression Testing**: Ensure new model meets quality gates
- **Synthetic vs. Real Performance**: Compare model performance on synthetic test data vs. real-world data
- **M4 Performance Analysis**:
  - Training time comparison across different settings
  - Memory utilization during training and inference
  - Model parameter count vs. performance curve
  - Inference latency on M4 for RoBERTa-base vs RoBERTa-large
- **Urgency Detection Focus**:
  - Evaluate specifically on ability to distinguish urgent vs. standard negative cases
  - Create dedicated test set with difficult edge cases for urgent scenarios

### 4.3 Qualitative Analysis

- **Error Analysis**: Review misclassified examples
- **Edge Case Testing**: Test with unusual but important scenarios
- **Confusion Pair Analysis**: Special focus on urgent negative vs. standard negative confusion
- **Sentiment Progression Tracking**: Evaluate model's ability to detect sentiment changes over conversation

## 5. Integration with Existing Pipeline

### 5.1 Model Registry Integration

- Add sentiment model to existing `model_pipeline.py` lifecycle management
- Version sentiment models separately from intent/entity models
- Track sentiment-specific performance metrics

### 5.2 Inference Pipeline Updates for M4 MacBook

```python
# In NLUInferencer.__init__
try:
    # Load the fine-tuned RoBERTa sentiment model
    self.sentiment_model_path = model_file_path("sentiment_model")

    # M4-optimized model loading
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    # Configure for efficient inference on M4
    torch_dtype = torch.float16  # Use mixed precision for faster inference

    self.sentiment_model = RobertaForSequenceClassification.from_pretrained(
        self.sentiment_model_path,
        torch_dtype=torch_dtype  # Enable mixed precision for faster inference
    )
    self.sentiment_model.to(self.device)
    self.sentiment_model.eval()

    self.sentiment_tokenizer = RobertaTokenizer.from_pretrained(
        self.sentiment_model_path
    )

    # Load sentiment mappings
    with open(model_file_path("sentiment_model/sentiment2id.json"), "r") as f:
        self.sentiment2id = json.load(f)
    self.id2sentiment = {v: k for k, v in self.sentiment2id.items()}

    print(f"INFO [NLUInferencer]: Sentiment model loaded successfully with M4 optimizations.")
except Exception as e:
    # Fallback to default if model isn't available
    print(f"WARNING [NLUInferencer]: Failed to load sentiment model: {e}")
    self.sentiment_model = None
```

- **M4-Optimized Inference**:
  - Expected inference latency: <30ms per request (vs. ~80-100ms with DistilBERT)
  - Can cache model predictions for common utterances
  - Use batch inference for analyzing multi-turn conversations
  - Consider model distillation for even faster performance

### 5.3 Response Adaptation

- **Enhanced Sentiment Tracking**:
  - Track sentiment history over 5-10 turns (viable with 36GB RAM)
  - Implement sentiment progression analytics
  - Create sentiment embeddings to visualize conversation tone over time
  - Use sentiment patterns to detect frustration escalation

- **Advanced Response Strategies**:
  - Dynamic response generation based on detected sentiment
  - Automatic priority escalation for urgent negative sentiment
  - Sentiment-aware conversation summarization
  - A/B test different response styles for various sentiment categories

## 6. Implementation Timeline

| Phase                                     | Duration      | Description                                                     |
| ----------------------------------------- | ------------- | --------------------------------------------------------------- |
| ChatGPT Data Generation                   | 2-3 days      | Generate data using ChatGPT, format and validate                |
| Data Processing & Analysis                | 1-2 days      | Clean, balance and analyze generated data                       |
| Model Training & Tuning                   | 2-3 days      | Train RoBERTa on M4 with optimal parameters                     |
| Evaluation & Benchmarking                 | 1-2 days      | Compare model performance metrics                               |
| Pipeline Integration                      | 2-3 days      | Integrate with existing codebase                                |
| Testing & Validation                      | 2-3 days      | End-to-end testing and refinement                               |
| **Total**                                 | **10-16 days** | **Approximately 2 weeks with ChatGPT data acceleration**       |

## 7. Success Criteria

1. **Performance Improvement**: ≥15% improvement in F1 score for urgent negative sentiment detection
2. **Latency Requirement**: <30ms per inference on M4 hardware
3. **Integration**: Seamless integration with existing pipeline
4. **User Experience**: Measurable improvement in appropriate handling of urgent situations
5. **Data Quality**: At least 5,000 high-quality ChatGPT-generated examples with balanced category distribution
6. **Accuracy Targets**:
   - 90%+ accuracy on urgent negative classification
   - 85%+ accuracy on standard negative classification
   - 80%+ accuracy on neutral and positive classification

## 8. Future Improvements

- **Multi-task Learning**: Combine sentiment with intent detection in a single model
- **Continuous Learning**: Implement feedback loop for model improvement
- **Explainability**: Add confidence scores and attribution for sentiment decisions
- **Synthetic Data Generation Pipeline**: Automate the process of generating and validating synthetic data
- **Sentiment Change Tracking**: Implement advanced tracking of sentiment shifts during conversations
- **Emotion Detection**: Extend beyond sentiment to detect specific emotions (frustrated, confused, satisfied)
- **Contextual Sentiment**: Consider conversation context when determining sentiment (beyond single-turn analysis)
- **M4-Specific Optimizations**:
  - Implement CoreML model conversion for native inference
  - Create on-device quantized models for deployment
  - Explore ensemble methods combining multiple sentiment models
  - Implement model pruning to improve speed while maintaining accuracy
