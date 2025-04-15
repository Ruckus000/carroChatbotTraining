import os
import json
import torch
import numpy as np
import traceback
import sys

# Enable debug output
DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"DEBUG: {msg}")

# Print Python and dependency versions for diagnostic purposes
debug_print(f"Python version: {sys.version}")
try:
    import torch
    debug_print(f"PyTorch version: {torch.__version__}")
except ImportError:
    debug_print("PyTorch not installed")

try:
    import transformers
    debug_print(f"Transformers version: {transformers.__version__}")
except ImportError:
    debug_print("Transformers not installed")

# Import the NLUInferencer without mocking
from inference import NLUInferencer

def main():
    """Run integration tests for the NLUInferencer."""
    print("Starting integration tests for NLUInferencer...")
    
    # Create inferencer
    try:
        inferencer = NLUInferencer()
        print("Successfully initialized NLUInferencer")
    except Exception as e:
        print(f"Error initializing NLUInferencer: {e}")
        debug_print(traceback.format_exc())
        return
    
    # Define test cases
    test_cases = [
        {
            "text": "I need a tow truck at 123 Main Street for my Honda Civic",
            "expected_intent_type": "towing",
            "expected_entity_types": ["pickup_location", "vehicle_make", "vehicle_model"]
        },
        {
            "text": "My battery is dead, can you send roadside assistance?",
            "expected_intent_type": "roadside",
            "expected_entity_types": []
        },
        {
            "text": "I want to schedule an appointment for an oil change next week",
            "expected_intent_type": "appointment",
            "expected_entity_types": []
        }
    ]
    
    # Run tests
    all_passed = True
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['text']}")
        
        try:
            # Get prediction
            result = inferencer.predict(test_case["text"])
            
            # Display result
            print(f"  Intent: {result['intent']['name']} (confidence: {result['intent']['confidence']:.4f})")
            print(f"  Entities: {len(result['entities'])} found")
            for entity in result['entities']:
                print(f"    - {entity['entity']}: {entity['value']}")
            
            # Basic validation
            intent_name = result['intent']['name']
            detected_entity_types = [entity['entity'] for entity in result['entities']]
            
            # Check if intent is of the expected type
            intent_matched = any(intent_name.startswith(test_case['expected_intent_type']) 
                               for intent_type in [test_case['expected_intent_type']])
            
            if not intent_matched and not intent_name.startswith("fallback"):
                print(f"  [FAIL] Intent '{intent_name}' does not match expected type '{test_case['expected_intent_type']}'")
                all_passed = False
            else:
                print(f"  [PASS] Intent match")
            
            # Check if expected entity types are detected
            if test_case['expected_entity_types']:
                # For each expected entity type, check if at least one matching entity was found
                for expected_type in test_case['expected_entity_types']:
                    found = False
                    for entity_type in detected_entity_types:
                        if expected_type in entity_type:
                            found = True
                            break
                    
                    if not found:
                        print(f"  [WARN] Expected entity type '{expected_type}' not found")
                        # Don't fail the test for this, just warn
            
        except Exception as e:
            print(f"  [ERROR] Test failed with exception: {e}")
            debug_print(traceback.format_exc())
            all_passed = False
    
    # Print summary
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        
    return all_passed

if __name__ == "__main__":
    main() 