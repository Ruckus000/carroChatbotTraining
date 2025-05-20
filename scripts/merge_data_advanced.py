# /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/merge_data_advanced.py
import json
import hashlib

def example_fingerprint(example):
    """Create a fingerprint of an example based on text, intent, and entities."""
    # Start with the text and intent
    text = example.get("text", "").lower().strip()
    intent = example.get("intent", "")

    # Sort entities by entity type and value for consistent comparison
    entities = sorted([
        f"{e.get('entity')}:{e.get('value')}"
        for e in example.get("entities", [])
    ])

    # Create a string representation
    fingerprint_str = f"{text}|{intent}|{','.join(entities)}"

    # Hash for efficiency in comparisons
    return hashlib.md5(fingerprint_str.encode()).hexdigest()

# Load the existing training data
with open("data/nlu_training_data.json", "r") as f:
    existing_data = json.load(f)

# Load the new training data
with open("new_training_data.json", "r") as f:
    new_data = json.load(f)

# Calculate fingerprints for existing data
existing_fingerprints = {example_fingerprint(ex): ex for ex in existing_data}

# Filter and add only unique examples
unique_new_examples = []
similar_examples = []
exact_duplicates = []

for example in new_data:
    fp = example_fingerprint(example)

    # Check for exact text match (strict duplicate)
    text_matches = [ex for ex in existing_data if ex["text"] == example["text"]]

    if text_matches:
        exact_duplicates.append(example)
    elif fp in existing_fingerprints:
        # Same fingerprint but different text = similar example
        similar_examples.append((example, existing_fingerprints[fp]))
    else:
        unique_new_examples.append(example)
        # Add to fingerprints to prevent duplicates within new_data
        existing_fingerprints[fp] = example

# Print detailed statistics
print(f"Existing data: {len(existing_data)} examples")
print(f"New data: {len(new_data)} examples")
print(f"Unique new examples: {len(unique_new_examples)} examples")
print(f"Exact duplicates: {len(exact_duplicates)} examples")
print(f"Similar examples (same intent/entities): {len(similar_examples)} examples")

# If specified, show details about similar examples
if similar_examples and len(similar_examples) <= 10:
    print("\nSimilar examples (might want to review):")
    for i, (new_ex, existing_ex) in enumerate(similar_examples):
        print(f"  {i+1}. NEW: \"{new_ex['text']}\" ({new_ex['intent']})")
        print(f"     OLD: \"{existing_ex['text']}\" ({existing_ex['intent']})")

# If there are no unique examples, exit early
if not unique_new_examples:
    print("No new unique examples to add. Exiting without changes.")
    exit(0)

# Ask for confirmation
confirm = input(f"\nAdd {len(unique_new_examples)} unique examples? (y/n): ")
if confirm.lower() != 'y':
    print("Operation cancelled.")
    exit(0)

# Create a backup of the original file
with open("data/nlu_training_data.json.bak", "w") as f:
    json.dump(existing_data, f, indent=2)

# Merge only the unique examples
merged_data = existing_data + unique_new_examples

# Save the merged data
with open("data/nlu_training_data.json", "w") as f:
    json.dump(merged_data, f, indent=2)

print(
    f"Merge completed successfully. Added {len(unique_new_examples)} unique examples."
    f"\nA backup of the original data was saved as 'data/nlu_training_data.json.bak'."
) 