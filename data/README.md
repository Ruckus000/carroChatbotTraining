# Sample Data for Chatbot Training

This directory contains sample conversation data for fine-tuning the DistilBERT-based chatbot system. The data covers three primary conversation flows: towing service, roadside assistance, and service appointment booking.

## Data Structure

The sample data in `sample_conversations.json` includes:

- **20 diverse examples** covering all primary flows
- **Entity variations** for each flow
- **Fallback examples** for out-of-domain queries
- **Clarification examples** for ambiguous requests

## Data Format

Each conversation example follows this JSON structure:

```json
{
  "flow": "towing",                     // Primary conversation flow
  "intent": "request_tow_location",     // Specific user intent
  "input": "I need a tow from...",      // User input text
  "response": "Got it! I'll arrange...", // Bot response
  "context": {"display_map": true},     // Additional context information
  "entities": [                         // Extracted entities
    {"entity": "pickup_location", "value": "123 Main Street"},
    {"entity": "destination", "value": "ABC Auto Shop"}
  ]
}
```

## Available Flows

1. **towing** - For tow truck requests
   - Intents: request_tow_basic, request_tow_location, request_tow_vehicle, request_tow_full, request_tow_urgent
   - Entities: pickup_location, destination, vehicle_year, vehicle_make, vehicle_model, urgency

2. **roadside** - For roadside assistance
   - Intents: request_roadside_basic, request_roadside_battery, request_roadside_tire, request_roadside_keys, request_roadside_fuel, request_roadside_full
   - Entities: service_type, pickup_location, vehicle_year, vehicle_make, vehicle_model

3. **appointment** - For service booking
   - Intents: book_service_basic, book_service_type, book_service_date, book_service_time, book_service_full
   - Entities: service_type, appointment_date, appointment_time, vehicle_year, vehicle_make, vehicle_model

4. **fallback** - For out-of-domain queries
   - Intents: out_of_domain

5. **clarification** - For ambiguous requests
   - Intents: ambiguous_request

## Using the Sample Data

You can use this data in two ways:

1. **Basic Training**: Use as-is for a baseline model
   ```bash
   python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output
   ```

2. **Data Augmentation**: Use with augmentation for a more robust model
   ```bash
   python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output --augment_data
   ```

3. **Full Pipeline**: Run the complete training and evaluation process
   ```bash
   python chatbot_training.py --input_data data/sample_conversations.json --output_dir ./output --augment_data --train_models --evaluate_models
   ```

## Extending the Sample Data

To build a production-quality chatbot, you should:

1. **Add more examples**: Aim for at least 100 examples per intent
2. **Diversify entities**: Include more vehicle makes/models, service types, locations
3. **Add entity variations**: Include different formats for dates, times, vehicle info
4. **Include edge cases**: Add examples with unusual phrasing or complex requests
5. **Add multi-turn conversations**: Include examples of follow-up questions and responses

The `utils.py` script includes a `create_conversation_data_template()` function that generates a template for creating new conversation examples.