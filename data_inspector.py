import json

def print_data_structure(filepath):
    """Inspect the structure of the training data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Found {len(data)} examples in {filepath}")
        
        # Check the first few examples
        for i, example in enumerate(data[:5]):
            print(f"\nExample {i+1}:")
            print(f"Text: {example['text']}")
            print(f"Intent: {example['intent']}")
            
            print(f"Entities type: {type(example['entities'])}")
            if 'entities' in example:
                if isinstance(example['entities'], list):
                    print(f"Entities (list of {len(example['entities'])} items):")
                    for j, entity in enumerate(example['entities']):
                        print(f"  Entity {j+1} type: {type(entity)}")
                        if isinstance(entity, dict):
                            print(f"    Keys: {entity.keys()}")
                        else:
                            print(f"    Value: {entity}")
                else:
                    print(f"Entities (non-list): {example['entities']}")
            else:
                print("No entities field found")
        
        # Check for inconsistent entity structures
        non_list_entities = [i for i, ex in enumerate(data) if not isinstance(ex.get('entities', []), list)]
        if non_list_entities:
            print(f"\nFound {len(non_list_entities)} examples with non-list entities at indices: {non_list_entities[:10]}...")
        
        missing_keys = []
        invalid_entity_format = []
        for i, example in enumerate(data):
            if 'entities' not in example:
                missing_keys.append(i)
                continue
                
            if not isinstance(example['entities'], list):
                continue  # Already counted above
                
            for j, entity in enumerate(example['entities']):
                if not isinstance(entity, dict) or 'entity' not in entity or 'value' not in entity:
                    invalid_entity_format.append((i, j))
                    break
                    
        if missing_keys:
            print(f"\nFound {len(missing_keys)} examples missing 'entities' key at indices: {missing_keys[:10]}...")
            
        if invalid_entity_format:
            print(f"\nFound {len(invalid_entity_format)} examples with invalid entity format at indices: {invalid_entity_format[:10]}...")
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in {filepath}")

if __name__ == "__main__":
    print_data_structure('data/nlu_training_data.json') 