#!/usr/bin/env python3
"""
Training data generator for NLU system.
This script helps generate new training examples from templates.
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any

# Entity Lists for slot filling
VEHICLE_MAKES = [
    "Honda", "Toyota", "Ford", "Chevrolet", "Nissan", "BMW", "Mercedes", 
    "Audi", "Hyundai", "Kia", "Subaru", "Volkswagen", "Lexus", "Jeep", 
    "Dodge", "Tesla", "Mazda", "Volvo", "Acura", "Infiniti"
]

VEHICLE_MODELS = {
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"],
    "Ford": ["F-150", "Escape", "Explorer", "Mustang", "Focus"],
    "Chevrolet": ["Silverado", "Malibu", "Equinox", "Tahoe", "Suburban"],
    "Nissan": ["Altima", "Rogue", "Sentra", "Maxima", "Pathfinder"],
    "BMW": ["3-Series", "5-Series", "X3", "X5", "7-Series"],
    "Mercedes": ["C-Class", "E-Class", "GLE", "S-Class", "GLC"],
    "Audi": ["A4", "Q5", "A6", "Q7", "A3"],
    "DEFAULT": ["Model S", "Model 3", "Wrangler", "Grand Cherokee", "Outback", 
                "Forester", "Jetta", "Passat", "Soul", "Optima", "Elantra", "Sonata"]
}

LOCATIONS = [
    "123 Main Street", "downtown", "the shopping mall", "Walmart parking lot",
    "near the gas station", "456 Oak Avenue", "the highway", "I-95 exit 23",
    "the intersection of First and Main", "my home", "work", "the office",
    "Central Plaza", "north side of town", "the airport", "the train station",
    "outside my apartment", "the university campus", "the hospital parking lot"
]

SERVICE_TYPES = [
    "oil change", "tire rotation", "brake service", "inspection", 
    "engine diagnostic", "tune-up", "fluid check", "air conditioning service", 
    "transmission service", "wheel alignment", "battery replacement", 
    "filter replacement", "brake pad replacement", "spark plug replacement"
]

ISSUES = [
    "flat tire", "dead battery", "won't start", "overheating", "strange noise",
    "check engine light", "locked out", "out of gas", "smoking engine", 
    "brake problem", "transmission problem", "electrical issue", "broken window",
    "alarm won't stop", "leaking fluid"
]

DATES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
    "tomorrow", "next week", "next Monday", "this weekend",
    "January 15", "February 20", "March 10", "April 5", "May 12"
]

TIMES = [
    "morning", "afternoon", "evening", "9am", "10am", "11am", "12pm",
    "1pm", "2pm", "3pm", "4pm", "5pm", "as soon as possible", "first thing"
]

# Templates by intent
TEMPLATES = {
    "towing_request_tow": [
        "I need a tow {for|to} {LOCATION}",
        "My {VEHICLE_MAKE} {VEHICLE_MODEL} needs to be towed {from|to} {LOCATION}",
        "Can I get a tow truck {at|to|near} {LOCATION}?",
        "My car broke down at {LOCATION}, I need a tow",
        "Need towing service for my {VEHICLE_MAKE} {VEHICLE_MODEL} at {LOCATION}",
        "Tow truck needed at {LOCATION} for a {VEHICLE_MAKE} {VEHICLE_MODEL}",
        "My {VEHICLE_MAKE} won't start, need a tow from {LOCATION}",
        "Requesting a tow for my {VEHICLE_MAKE} at {LOCATION} please"
    ],
    
    "roadside_request_battery": [
        "My battery is dead at {LOCATION}",
        "Need a jump start for my {VEHICLE_MAKE} {VEHICLE_MODEL} at {LOCATION}",
        "Battery died, I'm at {LOCATION} in a {VEHICLE_MAKE}",
        "My {VEHICLE_MAKE} {VEHICLE_MODEL} won't start, battery issue at {LOCATION}",
        "Can someone jump my car? I'm at {LOCATION}",
        "Need roadside assistance for a dead battery at {LOCATION}"
    ],
    
    "roadside_request_tire": [
        "I have a flat tire at {LOCATION}",
        "Need help changing a tire on my {VEHICLE_MAKE} at {LOCATION}",
        "Flat tire on my {VEHICLE_MAKE} {VEHICLE_MODEL}, I'm at {LOCATION}",
        "Can someone help with a flat? I'm stranded at {LOCATION}",
        "Need tire assistance at {LOCATION} for my {VEHICLE_MAKE}"
    ],
    
    "appointment_book_service": [
        "I want to schedule {a|an} {SERVICE_TYPE} for my {VEHICLE_MAKE} {VEHICLE_MODEL}",
        "Can I book an appointment for {SERVICE_TYPE} on {DATE} at {TIME}?",
        "Need to set up {SERVICE_TYPE} for {my|our} {VEHICLE_MAKE} on {DATE}",
        "I'd like to bring my {VEHICLE_MAKE} in for {SERVICE_TYPE}",
        "Booking request for {SERVICE_TYPE} on {DATE} at {TIME}",
        "Can I schedule my {VEHICLE_MAKE} {VEHICLE_MODEL} for {SERVICE_TYPE}?"
    ]
}

def expand_template(template: str) -> str:
    """
    Expand a template by replacing placeholders with random values.
    
    Args:
        template: Template string with placeholders like {VEHICLE_MAKE}
        
    Returns:
        Expanded template with placeholders replaced by values
    """
    # Handle choices like {a|an}
    while "{" in template and "|" in template and "}" in template:
        choice_start = template.find("{")
        choice_end = template.find("}", choice_start)
        
        if "|" in template[choice_start:choice_end]:
            choices = template[choice_start+1:choice_end].split("|")
            selected = random.choice(choices)
            template = template[:choice_start] + selected + template[choice_end+1:]
        else:
            break
    
    # Handle entity placeholders
    while "{VEHICLE_MAKE}" in template:
        make = random.choice(VEHICLE_MAKES)
        template = template.replace("{VEHICLE_MAKE}", make, 1)
        
    while "{VEHICLE_MODEL}" in template:
        # Try to match model to make if available
        if "makes" in locals():
            models = VEHICLE_MODELS.get(make, VEHICLE_MODELS["DEFAULT"])
        else:
            models = random.choice([v for v in VEHICLE_MODELS.values()])
        
        model = random.choice(models)
        template = template.replace("{VEHICLE_MODEL}", model, 1)
    
    template = template.replace("{LOCATION}", random.choice(LOCATIONS))
    template = template.replace("{SERVICE_TYPE}", random.choice(SERVICE_TYPES))
    template = template.replace("{ISSUE}", random.choice(ISSUES))
    template = template.replace("{DATE}", random.choice(DATES))
    template = template.replace("{TIME}", random.choice(TIMES))
    
    return template

def create_example(intent: str) -> Dict[str, Any]:
    """
    Create a complete example for a given intent.
    
    Args:
        intent: The intent name
        
    Returns:
        Dictionary with text, intent, and entities
    """
    if intent not in TEMPLATES:
        raise ValueError(f"No templates available for intent: {intent}")
    
    # Select a random template for this intent
    template = random.choice(TEMPLATES[intent])
    
    # Track which entities we'll use
    entities_to_extract = []
    if "{VEHICLE_MAKE}" in template:
        entities_to_extract.append("vehicle_make")
    if "{VEHICLE_MODEL}" in template:
        entities_to_extract.append("vehicle_model")
    if "{LOCATION}" in template:
        entities_to_extract.append("pickup_location")
    if "{SERVICE_TYPE}" in template:
        entities_to_extract.append("service_type") 
    if "{ISSUE}" in template:
        entities_to_extract.append("issue")
    if "{DATE}" in template:
        entities_to_extract.append("date")
    if "{TIME}" in template:
        entities_to_extract.append("time")
    
    # Generate entity values to use
    entity_values = {}
    if "vehicle_make" in entities_to_extract:
        entity_values["vehicle_make"] = random.choice(VEHICLE_MAKES)
    if "vehicle_model" in entities_to_extract:
        if "vehicle_make" in entity_values:
            models = VEHICLE_MODELS.get(entity_values["vehicle_make"], VEHICLE_MODELS["DEFAULT"])
        else:
            models = random.choice([v for v in VEHICLE_MODELS.values()])
        entity_values["vehicle_model"] = random.choice(models)
    if "pickup_location" in entities_to_extract:
        entity_values["pickup_location"] = random.choice(LOCATIONS)
    if "service_type" in entities_to_extract:
        entity_values["service_type"] = random.choice(SERVICE_TYPES)
    if "issue" in entities_to_extract:
        entity_values["issue"] = random.choice(ISSUES)
    if "date" in entities_to_extract:
        entity_values["date"] = random.choice(DATES)
    if "time" in entities_to_extract:
        entity_values["time"] = random.choice(TIMES)
    
    # Create the text by replacing placeholders with actual values
    text = template
    for entity_type, value in entity_values.items():
        if entity_type == "vehicle_make":
            text = text.replace("{VEHICLE_MAKE}", value)
        elif entity_type == "vehicle_model":
            text = text.replace("{VEHICLE_MODEL}", value)
        elif entity_type == "pickup_location":
            text = text.replace("{LOCATION}", value)
        elif entity_type == "service_type":
            text = text.replace("{SERVICE_TYPE}", value)
        elif entity_type == "issue":
            text = text.replace("{ISSUE}", value)
        elif entity_type == "date":
            text = text.replace("{DATE}", value)
        elif entity_type == "time":
            text = text.replace("{TIME}", value)
    
    # Handle choice options like {a|an}
    while "{" in text and "|" in text and "}" in text:
        choice_start = text.find("{")
        choice_end = text.find("}", choice_start)
        
        if "|" in text[choice_start:choice_end]:
            choices = text[choice_start+1:choice_end].split("|")
            selected = random.choice(choices)
            text = text[:choice_start] + selected + text[choice_end+1:]
    
    # Create the entities list
    entities = []
    for entity_type, value in entity_values.items():
        # Only add entities that actually appear in the text
        if value in text:
            entities.append({
                "entity": entity_type,
                "value": value
            })
    
    return {
        "text": text,
        "intent": intent,
        "entities": entities
    }

def generate_examples(count: int, output_file: str, append: bool = True):
    """
    Generate and save examples.
    
    Args:
        count: Number of examples to generate
        output_file: Path to save the examples
        append: Whether to append to existing file or create new one
    """
    # Generate examples
    examples = []
    intents = list(TEMPLATES.keys())
    
    for _ in range(count):
        intent = random.choice(intents)
        example = create_example(intent)
        examples.append(example)
    
    # Save or append to file
    if append and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if isinstance(existing_data, list):
                examples = existing_data + examples
            else:
                print(f"Warning: {output_file} does not contain a JSON array. Creating new file.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading existing file: {e}. Creating new file.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {count} examples. Total examples in {output_file}: {len(examples)}")
    
    # Print a few examples
    print("\nSample examples generated:")
    for i, example in enumerate(examples[-min(3, count):]):
        print(f"{i+1}. Text: {example['text']}")
        print(f"   Intent: {example['intent']}")
        print(f"   Entities: {example['entities']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NLU training examples")
    parser.add_argument("-n", "--count", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("-o", "--output", type=str, default="data/generated_examples.json", 
                        help="Output file path")
    parser.add_argument("--no-append", action="store_true", help="Don't append to existing file")
    
    args = parser.parse_args()
    
    generate_examples(args.count, args.output, not args.no_append) 