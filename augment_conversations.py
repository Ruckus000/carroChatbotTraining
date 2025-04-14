#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate noisy variations of conversation samples using data augmentation.
"""

import json
import copy
from typing import List, Dict, Any
from data_augmentation import add_advanced_noise

def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """Load conversation samples from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_conversations(conversations: List[Dict[str, Any]], file_path: str) -> None:
    """Save conversations to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

def generate_noisy_variations(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate three noisy variations of a conversation sample."""
    variations = []
    
    for i in range(3):  # Generate 3 variations
        variation = copy.deepcopy(conversation)
        noisy_input = add_advanced_noise(conversation['input'])
        
        # Only add the variation if it's different from the original
        if noisy_input != conversation['input']:
            variation['input'] = noisy_input
            variation['augmentation'] = 'noisy_variant'
            variations.append(variation)
    
    return variations

def main():
    # Load original conversations
    original_conversations = load_conversations('data/sample_conversations.json')
    
    # Generate variations for each conversation
    augmented_conversations = []
    augmented_conversations.extend(original_conversations)  # Add original conversations first
    
    for conversation in original_conversations:
        variations = generate_noisy_variations(conversation)
        augmented_conversations.extend(variations)
    
    # Save augmented dataset
    save_conversations(augmented_conversations, 'data/augmented_sample_conversations.json')
    
    # Print statistics
    print(f"Original conversations: {len(original_conversations)}")
    print(f"Total conversations after augmentation: {len(augmented_conversations)}")

if __name__ == '__main__':
    main() 