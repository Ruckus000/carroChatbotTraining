import json
import os
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score as seqeval_f1_score

def load_data(filepath):
    """Load training data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} examples from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in {filepath}")
        return []

def convert_text_entities_to_bio(text, entities):
    """Convert text and entities to word tokens with BIO tags"""
    # Split text into words
    words = text.split()
    # Initialize all tags as 'O'
    tags = ['O'] * len(words)
    
    for entity in entities:
        # Skip if entity doesn't have the required fields
        if not isinstance(entity, dict) or 'entity' not in entity or 'value' not in entity:
            continue
            
        entity_type = entity['entity']
        entity_value = entity['value']
        
        # Skip if entity_value is not a string
        if not isinstance(entity_value, str):
            continue
        
        # Find where the entity appears in the words
        entity_words = entity_value.split()
        entity_len = len(entity_words)
        
        for i in range(len(words) - entity_len + 1):
            # Try to match the entire phrase
            potential_match = ' '.join(words[i:i+entity_len])
            if potential_match.lower() == entity_value.lower():
                # Mark the first word as B-entity
                tags[i] = f'B-{entity_type}'
                # Mark subsequent words as I-entity
                for j in range(1, entity_len):
                    tags[i+j] = f'I-{entity_type}'
                break
    
    return words, tags

def standardize_data(data):
    """Ensure all examples have the expected format."""
    standardized = []
    for example in data:
        # Check required fields
        if 'text' not in example or 'intent' not in example:
            print(f"Skipping example missing required fields: {example}")
            continue
            
        # Ensure entities is a list
        entities = example.get('entities', [])
        if not isinstance(entities, list):
            entities = []
            
        # Ensure each entity has entity and value keys
        valid_entities = []
        for entity in entities:
            if isinstance(entity, dict) and 'entity' in entity and 'value' in entity:
                valid_entities.append({
                    'entity': entity['entity'],
                    'value': entity['value']
                })
                
        standardized.append({
            'text': example['text'],
            'intent': example['intent'],
            'entities': valid_entities
        })
        
    return standardized

def compute_intent_metrics(eval_pred):
    """Compute metrics for intent classification"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_entity_metrics(eval_pred):
    """Compute metrics for entity recognition"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Create lists to store true and predicted labels
    true_labels = []
    pred_labels = []
    
    # Convert ids to tag names and filter out -100 (padding tokens)
    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_seq.append(id2tag[labels[i][j]])
                pred_seq.append(id2tag[predictions[i][j]])
        
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    # Calculate metrics using seqeval
    report = classification_report(true_labels, pred_labels, output_dict=True)
    f1 = seqeval_f1_score(true_labels, pred_labels)
    
    return {
        'f1': f1,
        'precision': report['micro avg']['precision'],
        'recall': report['micro avg']['recall']
    }

def prepare_entity_dataset(examples, tokenizer, tag2id):
    """Prepare dataset for entity recognition."""
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i, (text, entities) in enumerate(zip(examples['text'], examples['entities'])):
        try:
            # Convert to BIO format
            words, tags = convert_text_entities_to_bio(text, entities)
            
            # Tokenize each word and align labels
            word_tokens = []
            word_label_ids = []
            
            for word, tag in zip(words, tags):
                # Handle empty words (shouldn't happen but just in case)
                if not word:
                    continue
                
                # Tokenize the word into subwords
                subwords = tokenizer.tokenize(word)
                if not subwords:  # Handle cases where tokenization returns empty
                    subwords = [tokenizer.unk_token]
                
                # Add the subwords and their label
                word_tokens.extend(subwords)
                
                # Get tag ID, default to O if not found
                tag_id = tag2id.get(tag, tag2id['O'])
                
                # Add the tag ID for the first subword
                word_label_ids.append(tag_id)
                
                # Add -100 for the remaining subwords (to be ignored in loss)
                word_label_ids.extend([-100] * (len(subwords) - 1))
            
            # Add CLS and SEP tokens
            encoded_inputs = tokenizer.encode_plus(
                word_tokens,
                is_split_into_words=False,  # Already tokenized
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Adjust labels for CLS and SEP tokens
            labels = [-100] + word_label_ids + [-100] * (len(encoded_inputs["input_ids"][0]) - len(word_label_ids) - 1)
            labels = labels[:len(encoded_inputs["input_ids"][0])]  # Truncate if needed
            
            tokenized_inputs["input_ids"].append(encoded_inputs["input_ids"][0])
            tokenized_inputs["attention_mask"].append(encoded_inputs["attention_mask"][0])
            tokenized_inputs["labels"].append(torch.tensor(labels))
        
        except Exception as e:
            print(f"Error preparing entity example {i}: {e}")
            # Skip this example
            continue
    
    # Ensure we have at least one example
    if len(tokenized_inputs["input_ids"]) == 0:
        raise ValueError("No valid examples after preprocessing. Check entity format.")
    
    # Convert lists to tensors
    tokenized_inputs["input_ids"] = torch.stack(tokenized_inputs["input_ids"])
    tokenized_inputs["attention_mask"] = torch.stack(tokenized_inputs["attention_mask"])
    tokenized_inputs["labels"] = torch.stack(tokenized_inputs["labels"])
    
    return tokenized_inputs

if __name__ == "__main__":
    # Set fixed random state for reproducibility
    RANDOM_STATE = 42
    
    # Load data
    data = load_data('data/nlu_training_data.json')
    if not data:
        print("No data loaded. Exiting.")
        exit(1)
    
    # Standardize the data format
    data = standardize_data(data)
    print(f"Standardized data: {len(data)} examples")
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Split data into {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Prepare intent classification data
    all_intents = sorted(list(set(example['intent'] for example in train_data)))
    intent2id = {intent: i for i, intent in enumerate(all_intents)}
    id2intent = {i: intent for intent, i in intent2id.items()}
    
    # Save intent mappings
    os.makedirs('./trained_nlu_model/intent_model', exist_ok=True)
    try:
        with open('./trained_nlu_model/intent_model/intent2id.json', 'w', encoding='utf-8') as f:
            json.dump(intent2id, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error saving intent mappings: {e}")
    
    # Prepare entity recognition data
    # Get all unique entity types from training data
    entity_types = set()
    for example in train_data:
        for entity in example['entities']:
            entity_types.add(entity['entity'])
    
    # Create BIO tags
    bio_tags = ['O']
    for entity_type in sorted(entity_types):
        bio_tags.extend([f'B-{entity_type}', f'I-{entity_type}'])
    
    tag2id = {tag: i for i, tag in enumerate(bio_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    
    # Save entity mappings
    os.makedirs('./trained_nlu_model/entity_model', exist_ok=True)
    try:
        with open('./trained_nlu_model/entity_model/tag2id.json', 'w', encoding='utf-8') as f:
            json.dump(tag2id, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error saving entity mappings: {e}")
    
    # Create datasets for intent classification
    intent_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_texts = [example['text'] for example in train_data]
    train_intent_ids = [intent2id[example['intent']] for example in train_data]
    
    val_texts = [example['text'] for example in val_data]
    val_intent_ids = []
    for example in val_data:
        intent = example['intent']
        # Handle unseen intents in validation set
        if intent not in intent2id:
            print(f"Warning: Unseen intent '{intent}' in validation set. Mapping to fallback.")
            # Map to a fallback intent ID if it exists, or to the first intent as default
            fallback_id = intent2id.get('fallback_out_of_domain', 0)
            val_intent_ids.append(fallback_id)
        else:
            val_intent_ids.append(intent2id[intent])
    
    # Tokenize for intent classification
    train_intent_encodings = intent_tokenizer(train_texts, truncation=True, padding=True)
    val_intent_encodings = intent_tokenizer(val_texts, truncation=True, padding=True)
    
    # Create datasets
    train_intent_dataset = Dataset.from_dict({
        'input_ids': train_intent_encodings['input_ids'],
        'attention_mask': train_intent_encodings['attention_mask'],
        'labels': train_intent_ids
    })
    
    val_intent_dataset = Dataset.from_dict({
        'input_ids': val_intent_encodings['input_ids'],
        'attention_mask': val_intent_encodings['attention_mask'],
        'labels': val_intent_ids
    })
    
    # Create datasets for entity recognition
    entity_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create simple Dataset objects first
    train_entity_raw_data = {
        'text': [example['text'] for example in train_data],
        'entities': [example['entities'] for example in train_data]
    }
    
    val_entity_raw_data = {
        'text': [example['text'] for example in val_data],
        'entities': [example['entities'] for example in val_data]
    }
    
    # Prepare entity datasets
    train_entity_features = prepare_entity_dataset(train_entity_raw_data, entity_tokenizer, tag2id)
    val_entity_features = prepare_entity_dataset(val_entity_raw_data, entity_tokenizer, tag2id)
    
    train_entity_dataset = Dataset.from_dict({
        'input_ids': train_entity_features['input_ids'].numpy().tolist(),
        'attention_mask': train_entity_features['attention_mask'].numpy().tolist(),
        'labels': train_entity_features['labels'].numpy().tolist()
    })
    
    val_entity_dataset = Dataset.from_dict({
        'input_ids': val_entity_features['input_ids'].numpy().tolist(),
        'attention_mask': val_entity_features['attention_mask'].numpy().tolist(),
        'labels': val_entity_features['labels'].numpy().tolist()
    })
    
    # Load models
    try:
        intent_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(intent2id),
            id2label=id2intent,
            label2id=intent2id
        )
    except Exception as e:
        print(f"Error loading intent model: {e}")
        exit(1)
    
    try:
        entity_model = DistilBertForTokenClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(tag2id),
            id2label=id2tag,
            label2id=tag2id
        )
    except Exception as e:
        print(f"Error loading entity model: {e}")
        exit(1)
    
    # Configure training arguments
    intent_training_args = TrainingArguments(
        output_dir='./trained_nlu_model/intent_model_checkpoints',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./trained_nlu_model/intent_logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        use_cpu=True
    )
    
    entity_training_args = TrainingArguments(
        output_dir='./trained_nlu_model/entity_model_checkpoints',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./trained_nlu_model/entity_logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        use_cpu=True
    )
    
    # Create trainers
    intent_trainer = Trainer(
        model=intent_model,
        args=intent_training_args,
        train_dataset=train_intent_dataset,
        eval_dataset=val_intent_dataset,
        compute_metrics=compute_intent_metrics
    )
    
    entity_trainer = Trainer(
        model=entity_model,
        args=entity_training_args,
        train_dataset=train_entity_dataset,
        eval_dataset=val_entity_dataset,
        compute_metrics=compute_entity_metrics
    )
    
    # Train models
    print("Training intent model...")
    try:
        intent_trainer.train()
    except Exception as e:
        print(f"Error during intent model training: {e}")
        exit(1)
    
    print("Training entity model...")
    try:
        entity_trainer.train()
    except Exception as e:
        print(f"Error during entity model training: {e}")
        exit(1)
    
    # Save models
    print("Saving intent model...")
    try:
        intent_model.save_pretrained('./trained_nlu_model/intent_model')
        intent_tokenizer.save_pretrained('./trained_nlu_model/intent_model')
    except Exception as e:
        print(f"Error saving intent model: {e}")
    
    print("Saving entity model...")
    try:
        entity_model.save_pretrained('./trained_nlu_model/entity_model')
        entity_tokenizer.save_pretrained('./trained_nlu_model/entity_model')
    except Exception as e:
        print(f"Error saving entity model: {e}")
    
    print("Training complete!") 