#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for data augmentation to enhance chatbot training data.
Includes noise injection, entity variations, and generation of special test cases.
"""

import random
import re
import copy
import string
from typing import List, Dict, Any, Union

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
        # Generate variations based on intent and flow
        variations = generate_variations(example['input'], example['intent'], example['flow'])
        
        for var in variations:
            new_example = copy.deepcopy(example)
            new_example['input'] = var
            conversation_extensions.append(new_example)
        
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

def generate_variations(input_text: str, intent: str, flow: str) -> List[str]:
    """Generate variations for specific intents and flows"""
    variations = []
    
    if flow == "towing" and intent == "request_tow_basic":
        variations = [
            "I need to get my car towed.",
            "My vehicle needs to be towed.",
            "Can you help tow my car?",
            "I'd like to arrange a tow service.",
            "Is it possible to get a tow truck?"
        ]
    
    elif flow == "roadside" and intent == "request_roadside_basic":
        variations = [
            "I need help with my car.",
            "Can you send roadside assistance?",
            "My vehicle is having issues and needs help.",
            "I'd like to get roadside help.",
            "Need someone to come help with my car."
        ]
    
    elif flow == "appointment" and intent == "book_service_basic":
        variations = [
            "I want to schedule car service.",
            "Need to make an appointment for car maintenance.",
            "Looking to book my car for service.",
            "Can I schedule my car to be serviced?",
            "Want to set up an appointment for my vehicle."
        ]
    
    return variations

def add_advanced_noise(text: str) -> str:
    """Apply various realistic noise patterns to text"""
    if random.random() > 0.7:  # Only apply 30% of the time
        return text
    
    noise_type = random.choice([
        "typo", "omission", "swapping", "spacing", 
        "capitalization", "slang", "abbreviation", "punctuation"
    ])
    
    if noise_type == "typo":
        # Replace a random character with an adjacent keyboard character
        if len(text) < 3:
            return text
        
        keyboard_adjacency = {
            'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
            'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn',
            'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edfgt', 's': 'qazxdcw', 't': 'rfghy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx'
        }
        
        char_idx = random.randint(0, len(text) - 1)
        char = text[char_idx].lower()
        
        if char in keyboard_adjacency:
            replacement = random.choice(keyboard_adjacency[char])
            if text[char_idx].isupper():
                replacement = replacement.upper()
            text = text[:char_idx] + replacement + text[char_idx+1:]
    
    elif noise_type == "omission":
        # Randomly omit a character
        if len(text) < 3:
            return text
            
        char_idx = random.randint(0, len(text) - 1)
        text = text[:char_idx] + text[char_idx+1:]
    
    elif noise_type == "swapping":
        # Swap two adjacent characters
        if len(text) < 3:
            return text
            
        char_idx = random.randint(0, len(text) - 2)
        text = text[:char_idx] + text[char_idx+1] + text[char_idx] + text[char_idx+2:]
    
    elif noise_type == "spacing":
        # Add or remove spaces
        words = text.split()
        if len(words) < 2:
            return text
            
        if random.random() > 0.5:  # Remove a space
            join_idx = random.randint(0, len(words) - 2)
            words[join_idx] = words[join_idx] + words[join_idx + 1]
            words.pop(join_idx + 1)
        else:  # Add an extra space
            word_idx = random.randint(0, len(words) - 1)
            if len(words[word_idx]) < 3:
                return text
                
            split_point = random.randint(1, len(words[word_idx]) - 1)
            words[word_idx] = words[word_idx][:split_point] + " " + words[word_idx][split_point:]
        
        text = " ".join(words)
    
    elif noise_type == "capitalization":
        # Random capitalization
        words = text.split()
        if not words:
            return text
            
        word_idx = random.randint(0, len(words) - 1)
        if random.random() > 0.5:  # ALL CAPS
            words[word_idx] = words[word_idx].upper()
        else:  # First letter only
            if words[word_idx]:
                words[word_idx] = words[word_idx][0].upper() + words[word_idx][1:]
        
        text = " ".join(words)
    
    elif noise_type == "slang":
        # Replace with common slang/text shortcuts
        slang_dict = {
            "please": "pls",
            "you": "u",
            "for": "4",
            "to": "2",
            "be": "b",
            "are": "r",
            "and": "&",
            "at": "@",
            "with": "w/",
            "without": "w/o",
            "appointment": "appt",
            "service": "svc",
            "vehicle": "car",
            "assistance": "help",
            "maintenance": "maint"
        }
        
        words = text.split()
        if not words:
            return text
            
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in slang_dict and random.random() > 0.5:
                words[i] = slang_dict[lower_word]
        
        text = " ".join(words)
    
    elif noise_type == "abbreviation":
        # Use common abbreviations
        abbrev_dict = {
            "appointment": "apt",
            "schedule": "sched",
            "vehicle": "veh",
            "transmission": "trans",
            "maintenance": "maint",
            "diagnostics": "diag",
            "assistance": "assist",
            "battery": "batt",
            "automatic": "auto",
            "immediately": "ASAP"
        }
        
        words = text.split()
        if not words:
            return text
            
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in abbrev_dict and random.random() > 0.5:
                words[i] = abbrev_dict[lower_word]
        
        text = " ".join(words)
    
    elif noise_type == "punctuation":
        # Add or remove punctuation
        if random.random() > 0.5:  # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
        else:  # Add extra punctuation
            punctuation = "!?.,;"
            insert_idx = random.randint(0, len(text) - 1)
            text = text[:insert_idx] + random.choice(punctuation) + text[insert_idx:]
    
    return text

def create_entity_variations(original_example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create variations with different entity patterns"""
    variations = []
    
    # Only proceed if there are entities to work with
    if not original_example.get('entities', []):
        return []
    
    # 1. Incomplete entity information
    if len(original_example['entities']) >= 2:
        incomplete = copy.deepcopy(original_example)
        # Remove a random entity
        entities_to_remove = random.sample(incomplete['entities'], 1)
        incomplete['entities'] = [e for e in incomplete['entities'] if e not in entities_to_remove]
        
        # Update the input text to reflect the missing entity
        missing_entity = entities_to_remove[0]
        input_text = incomplete['input']
        
        # Check that input_text is a string and missing_entity['value'] is a string
        if isinstance(input_text, str) and isinstance(missing_entity.get('value'), str):
            # Simple approach: if the entity value appears in the text, try to remove it
            if missing_entity['value'] in input_text:
                input_text = input_text.replace(missing_entity['value'], "")
                input_text = re.sub(r'\s+', ' ', input_text).strip()  # Clean up extra spaces
                incomplete['input'] = input_text
                
                # Update context to indicate partial information
                if 'context' in incomplete and isinstance(incomplete['context'], dict):
                    incomplete['context']['partial_info'] = True
                
                # Update response to ask for the missing information
                if missing_entity['entity'] == 'vehicle_make':
                    incomplete['response'] = "Can you tell me what make your vehicle is?"
                elif missing_entity['entity'] == 'vehicle_model':
                    incomplete['response'] = "What model is your vehicle?"
                elif missing_entity['entity'] == 'vehicle_year':
                    incomplete['response'] = "What year is your vehicle?"
                elif 'location' in missing_entity['entity']:
                    incomplete['response'] = "Could you provide the location information?"
                
                variations.append(incomplete)
    
    # 2. Out-of-order entity information
    if len(original_example['input'].split()) > 4:  # Only for longer inputs
        reordered = copy.deepcopy(original_example)
        input_words = reordered['input'].split()
        
        # Shuffle the middle portion of the sentence
        if len(input_words) > 5:
            middle_start = len(input_words) // 3
            middle_end = 2 * len(input_words) // 3
            
            middle_section = input_words[middle_start:middle_end]
            random.shuffle(middle_section)
            
            reordered['input'] = " ".join(
                input_words[:middle_start] + middle_section + input_words[middle_end:]
            )
            
            variations.append(reordered)
    
    # 3. Mixed format for entities
    if any(e['entity'] == 'vehicle_year' for e in original_example['entities']):
        mixed_format = copy.deepcopy(original_example)
        input_text = mixed_format['input']
        
        # Ensure input_text is a string
        if isinstance(input_text, str):
            # Find year mentions and convert format (2020 -> '20, etc.)
            for entity in mixed_format['entities']:
                if entity['entity'] == 'vehicle_year' and isinstance(entity.get('value'), str) and entity['value'].isdigit():
                    old_year = entity['value']
                    if len(old_year) == 4:  # Full year format (e.g., 2020)
                        new_year = "'" + old_year[2:]  # Convert to '20
                        input_text = input_text.replace(old_year, new_year)
                        entity['value'] = new_year  # Update entity value
            
            mixed_format['input'] = input_text
            variations.append(mixed_format)
    
    return variations

def generate_mixed_intent_examples() -> List[Dict[str, Any]]:
    """Create examples where user mixes multiple intents or domains"""
    return [
        {
            "flow": "clarification",
            "intent": "mixed_tow_service",
            "input": "I need a tow truck and then an oil change appointment.",
            "response": "I can help with both. Let's start with the tow truck request. Where should they pick up your vehicle, and where would you like it towed?",
            "context": {"mixed_intent": True, "primary_flow": "towing", "secondary_flow": "appointment"},
            "entities": []
        },
        {
            "flow": "clarification",
            "intent": "mixed_roadside_weather",
            "input": "My car won't start and it's going to rain soon.",
            "response": "I'll focus on getting your car started. What issue are you experiencing - is it the battery, ignition, or something else?",
            "context": {"mixed_intent": True, "primary_flow": "roadside", "contains_out_of_domain": True},
            "entities": [{"entity": "issue_type", "value": "won't start"}]
        },
        {
            "flow": "clarification",
            "intent": "mixed_appointment_query",
            "input": "I need an oil change and how much does it cost?",
            "response": "I'd be happy to book an oil change for you. The cost will depend on your vehicle type and the oil used. Could you tell me your vehicle's make, model, and year?",
            "context": {"mixed_intent": True, "primary_flow": "appointment", "contains_query": True},
            "entities": [{"entity": "service_type", "value": "oil change"}]
        },
        {
            "flow": "clarification",
            "intent": "tow_with_passengers",
            "input": "I need a tow and a ride for myself and 3 passengers.",
            "response": "I can arrange a tow truck for your vehicle. For passenger transportation, you might need a taxi or rideshare service. Would you like me to focus on the tow request first?",
            "context": {"mixed_intent": True, "primary_flow": "towing", "contains_out_of_domain": True},
            "entities": [{"entity": "passengers", "value": "4"}]
        }
    ]

def generate_extreme_test_cases() -> List[Dict[str, Any]]:
    """Generate particularly challenging test cases"""
    return [
        # Very terse inputs
        {
            "flow": "towing",
            "intent": "request_tow_basic",
            "input": "tow now",
            "response": "I'll help arrange a tow truck. Where should they pick up your vehicle and where should it be towed?",
            "context": {"display_map": True, "is_terse": True},
            "entities": []
        },
        # Run-on sentences with multiple topics
        {
            "flow": "clarification",
            "intent": "multiple_requests_run_on",
            "input": "my car broke down need a tow to the nearest garage it's making a weird noise and smoking plus I need to schedule an oil change for next week if possible",
            "response": "I understand you're in a difficult situation. Let's focus first on your immediate need - the tow for your broken down vehicle. Where are you located right now?",
            "context": {"is_run_on": True, "primary_flow": "towing", "secondary_flow": "appointment"},
            "entities": [{"entity": "issue_type", "value": "broke down"}]
        },
        # Heavy use of slang/text-speak
        {
            "flow": "roadside",
            "intent": "roadside_specific_service",
            "input": "batt dead @ walmart pls hlp asap",
            "response": "I understand your battery is dead at Walmart and you need urgent help. I'll arrange for battery jump start service. Could you confirm which Walmart location you're at?",
            "context": {"service_type": "battery_jump", "is_terse": True, "priority": "high"},
            "entities": [
                {"entity": "service_type", "value": "battery_jump"},
                {"entity": "location", "value": "walmart"}
            ]
        },
        # Grammatically incorrect or word salad
        {
            "flow": "clarification",
            "intent": "grammatically_incorrect",
            "input": "car no go need fix today please",
            "response": "I understand your car isn't working and you need it fixed today. Could you tell me more about what's happening with your car?",
            "context": {"needs_clarification": True},
            "entities": []
        },
        # Mixed languages
        {
            "flow": "clarification",
            "intent": "mixed_language",
            "input": "Mi carro needs a tow por favor",
            "response": "I see you need a tow truck for your car. Where should they pick up your vehicle and where should it be towed?",
            "context": {"display_map": True, "is_multilingual": True},
            "entities": []
        }
    ]
