#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for data processing and file handling.
"""

import json
import os
from typing import List, Dict, Any, Union, Optional

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing conversation data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save dataset to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path where the file will be saved
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def prepare_for_distilbert(dataframe: Any, task_type: str = "intent") -> List[Dict[str, Any]]:
    """
    Format data for DistilBERT fine-tuning based on task type.
    
    Args:
        dataframe: DataFrame containing conversation data
        task_type: Type of task ("intent", "flow", "entity", "fallback", "clarification")
        
    Returns:
        List of dictionaries formatted for the specified task
    """
    formatted_data = []
    
    if task_type == "intent":
        # For intent classification
        for _, row in dataframe.iterrows():
            formatted_data.append({
                "text": row['input'],
                "label": row['intent'],
                "flow": row['flow']
            })
    
    elif task_type == "flow":
        # For flow classification (which conversation flow)
        for _, row in dataframe.iterrows():
            formatted_data.append({
                "text": row['input'],
                "label": row['flow']
            })
    
    elif task_type == "entity":
        # For entity extraction
        for _, row in dataframe.iterrows():
            if len(row.get('entities', [])) > 0:
                formatted_data.append({
                    "text": row['input'],
                    "entities": row['entities']
                })
    
    elif task_type == "fallback":
        # Special task: detect if a query needs a fallback response
        for _, row in dataframe.iterrows():
            is_fallback = row['flow'] == 'fallback'
            formatted_data.append({
                "text": row['input'],
                "label": "fallback" if is_fallback else "not_fallback"
            })
    
    elif task_type == "clarification":
        # Special task: detect if a query needs clarification
        for _, row in dataframe.iterrows():
            needs_clarification = (
                row['flow'] == 'clarification' or 
                (isinstance(row.get('context', {}), dict) and 
                 row.get('context', {}).get('needs_clarification', False))
            )
            formatted_data.append({
                "text": row['input'],
                "label": "needs_clarification" if needs_clarification else "clear_intent"
            })
    
    elif task_type == "dialog":
        # For sequence-to-sequence dialog modeling
        for _, row in dataframe.iterrows():
            formatted_data.append({
                "input_text": row['input'],
                "target_text": row['response'],
                "context": row.get('context', {})
            })
    
    return formatted_data

def filter_by_flow(data: List[Dict[str, Any]], flow: str) -> List[Dict[str, Any]]:
    """
    Filter data to only include examples from a specific flow.
    
    Args:
        data: List of data points
        flow: Flow name to filter by
        
    Returns:
        Filtered list of data points
    """
    return [item for item in data if item.get('flow') == flow]

def convert_to_bio_tags(text: str, entities: List[Dict[str, Any]]) -> List[str]:
    """
    Convert entity annotations to BIO tagging format.
    
    Args:
        text: Input text
        entities: List of entity dictionaries with 'entity' and 'value' keys
        
    Returns:
        List of BIO tags for each token in the text
    """
    tokens = text.split()
    bio_tags = ['O'] * len(tokens)
    
    for entity in entities:
        entity_type = entity['entity']
        entity_value = str(entity['value'])
        
        # Simple token-based approach (not handling subword tokens)
        entity_tokens = entity_value.split()
        
        for i in range(len(tokens) - len(entity_tokens) + 1):
            match = True
            for j in range(len(entity_tokens)):
                if tokens[i + j].lower() != entity_tokens[j].lower():
                    match = False
                    break
            
            if match:
                # Mark beginning of entity
                bio_tags[i] = f'B-{entity_type}'
                
                # Mark inside of entity
                for j in range(1, len(entity_tokens)):
                    bio_tags[i + j] = f'I-{entity_type}'
    
    return bio_tags

def prepare_entity_extraction_data(dataframe: Any) -> List[Dict[str, Any]]:
    """
    Prepare data specifically for token classification (entity extraction).
    
    Args:
        dataframe: DataFrame containing conversation data
        
    Returns:
        List of dictionaries formatted for token classification
    """
    formatted_data = []
    
    for _, row in dataframe.iterrows():
        if len(row.get('entities', [])) > 0:
            tokens = row['input'].split()
            bio_tags = convert_to_bio_tags(row['input'], row['entities'])
            
            formatted_data.append({
                "tokens": tokens,
                "bio_tags": bio_tags,
                "text": row['input']
            })
    
    return formatted_data

def get_unique_entity_types(dataframe: Any) -> List[str]:
    """
    Get a list of all unique entity types in the dataset.
    
    Args:
        dataframe: DataFrame containing conversation data
        
    Returns:
        List of unique entity types
    """
    entity_types = set()
    
    for _, row in dataframe.iterrows():
        for entity in row.get('entities', []):
            entity_types.add(entity['entity'])
    
    return sorted(list(entity_types))

def create_bio_label_map(entity_types: List[str]) -> Dict[str, int]:
    """
    Create a mapping from BIO tags to integer IDs.
    
    Args:
        entity_types: List of entity types
        
    Returns:
        Dictionary mapping BIO tags to integer IDs
    """
    label_map = {"O": 0}  # Outside tag
    
    idx = 1
    for entity_type in entity_types:
        label_map[f"B-{entity_type}"] = idx
        idx += 1
        label_map[f"I-{entity_type}"] = idx
        idx += 1
    
    return label_map

def create_conversation_data_template() -> Dict[str, Any]:
    """
    Create a template for new conversation data.
    
    Returns:
        Dictionary template for conversation data
    """
    return {
        "flow": "",
        "intent": "",
        "input": "",
        "response": "",
        "context": {},
        "entities": []
    }

def merge_conversation_datasets(datasets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge multiple conversation datasets into one.
    
    Args:
        datasets: List of conversation datasets
        
    Returns:
        Merged conversation dataset
    """
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    
    return merged

def validate_conversation_data(conversations: List[Dict[str, Any]]) -> List[str]:
    """
    Validate conversation data for required fields and format.
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        List of validation error messages, empty if no errors
    """
    errors = []
    required_fields = ["flow", "intent", "input", "response", "entities"]
    
    for i, conv in enumerate(conversations):
        # Check required fields
        for field in required_fields:
            if field not in conv:
                errors.append(f"Conversation {i}: Missing required field '{field}'")
        
        # Validate entities format
        if "entities" in conv:
            for j, entity in enumerate(conv["entities"]):
                if "entity" not in entity:
                    errors.append(f"Conversation {i}, Entity {j}: Missing 'entity' field")
                if "value" not in entity:
                    errors.append(f"Conversation {i}, Entity {j}: Missing 'value' field")
    
    return errors

def create_initial_conversation_data() -> List[Dict[str, Any]]:
    """
    Create initial conversation data for testing.
    
    Returns:
        List of sample conversation dictionaries
    """
    return [
        {
            "flow": "towing",
            "intent": "request_tow_basic",
            "input": "I need a tow truck.",
            "response": "I can help with that! Where should they pick up your vehicle and where should it be towed?",
            "context": {"display_map": True},
            "entities": []
        },
        {
            "flow": "roadside",
            "intent": "request_roadside_basic",
            "input": "I need roadside assistance.",
            "response": "I can help with that! Can you tell me what issue you're experiencing with your vehicle?",
            "context": {"display_options": ["Battery", "Flat Tire", "Keys Locked", "Out of Fuel", "Won't Start", "Other"]},
            "entities": []
        },
        {
            "flow": "appointment",
            "intent": "book_service_basic",
            "input": "I need to book service for my car.",
            "response": "Got it! Let's get your appointment set up. What kind of service does your car need?",
            "context": {"display_options": ["Oil Change", "Tire Rotation", "Brake Check", "Diagnostic", "Regular Maintenance", "Not Sure"]},
            "entities": []
        }
    ]
