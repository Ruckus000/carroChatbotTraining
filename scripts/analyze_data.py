import json

# Load the data
data = json.load(open("data/nlu_training_data.json"))

# Count intents
intents = {}
for item in data:
    intents[item["intent"]] = intents.get(item["intent"], 0) + 1

# Count entities
entities = {}
for item in data:
    for entity in item.get("entities", []):
        entity_type = entity.get("entity")
        if entity_type:
            entities[entity_type] = entities.get(entity_type, 0) + 1

# Print results
print("=== Intent Distribution ===")
print(f"Total examples: {len(data)}")
print(f"Unique intents: {len(intents)}\n")
for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
    print(f"{intent}: {count}")

print("\n=== Entity Distribution ===")
print(f"Unique entity types: {len(entities)}\n")
for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True):
    print(f"{entity}: {count}")
