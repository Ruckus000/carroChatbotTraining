# /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/merge_data_unique.py
import json

# Load the existing training data
with open("data/nlu_training_data.json", "r") as f:
    existing_data = json.load(f)

# Load the new training data
with open("new_training_data.json", "r") as f:
    new_data = json.load(f)

# Extract existing example texts for faster comparison
existing_texts = set(example["text"] for example in existing_data)

# Filter out duplicates
unique_new_examples = []
duplicates = []

for example in new_data:
    if example["text"] in existing_texts:
        duplicates.append(example["text"])
    else:
        unique_new_examples.append(example)
        existing_texts.add(example["text"])  # Add to set to prevent duplicates within new_data

# Print statistics
print(f"Existing data: {len(existing_data)} examples")
print(f"New data: {len(new_data)} examples")
print(f"Unique new examples: {len(unique_new_examples)} examples")
print(f"Duplicates skipped: {len(duplicates)} examples")

if duplicates:
    print("\nFirst 5 duplicate examples (skipped):")
    for i, text in enumerate(duplicates[:5]):
        print(f"  {i+1}. {text}")

# If there are no unique examples, exit early
if not unique_new_examples:
    print("No new unique examples to add. Exiting without changes.")
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