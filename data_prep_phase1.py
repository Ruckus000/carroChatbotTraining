import json
import os

def consolidate_training_data():
    try:
        # Load source data
        with open('data/sample_conversations.json', 'r') as f:
            try:
                source_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return

        # Transform data
        transformed = []
        for example in source_data:
            try:
                # Create flattened intent
                flow = example.get('flow', 'fallback')
                intent = example.get('intent', 'out_of_domain')
                flattened_intent = f"{flow}_{intent}".lower()

                # Copy entities
                entities = example.get('entities', [])
                if not isinstance(entities, list):
                    entities = []

                transformed.append({
                    'text': str(example['input']),
                    'intent': flattened_intent,
                    'entities': entities
                })
            except KeyError as e:
                print(f"Skipping invalid example: {e}")
                continue

        # Add fallback examples if missing
        fallback_example = {
            'text': "What's the weather like?",
            'intent': 'fallback_out_of_scope_weather',
            'entities': []
        }
        transformed.append(fallback_example)

        # Save consolidated data
        os.makedirs('data', exist_ok=True)
        try:
            with open('data/nlu_training_data.json', 'w') as f:
                json.dump(transformed, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error writing output file: {e}")

    except FileNotFoundError as e:
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    consolidate_training_data() 