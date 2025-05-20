import json

# Load the existing training data
with open("data/nlu_training_data.json", "r") as f:
    existing_data = json.load(f)

# Load the new training data
with open("new_training_data.json", "r") as f:
    new_data = json.load(f)

# Merge the data
merged_data = existing_data + new_data

# Print statistics
print(f"Existing data: {len(existing_data)} examples")
print(f"New data: {len(new_data)} examples")
print(f"Merged data: {len(merged_data)} examples")

# Create a backup of the original file
with open("data/nlu_training_data.json.bak", "w") as f:
    json.dump(existing_data, f, indent=2)

# Save the merged data
with open("data/nlu_training_data.json", "w") as f:
    json.dump(merged_data, f, indent=2)

print(
    "Merge completed successfully. A backup of the original data was saved as 'data/nlu_training_data.json.bak'."
)
