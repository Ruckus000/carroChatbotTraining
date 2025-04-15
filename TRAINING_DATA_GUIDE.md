# Training Data Expansion Guide

This guide provides detailed strategies for expanding the NLU training data to improve model performance.

## Current Data Analysis

The existing training data in `data/nlu_training_data.json` contains examples covering these main categories:

1. **Towing Requests**
2. **Roadside Assistance**
3. **Appointment Scheduling**
4. **Fallback/Out-of-Domain**

## Systematic Expansion Approach

### 1. Intent-Specific Templates

Below are templates for generating new examples for each intent category.

#### Towing Request Templates

```
I need a tow [for/to] [LOCATION]
My [VEHICLE_MAKE] [VEHICLE_MODEL] needs to be towed [from/to] [LOCATION]
Can I get a tow truck [at/to/near] [LOCATION]?
My car broke down at [LOCATION], I need a tow
Need towing service for my [VEHICLE_MAKE] [VEHICLE_MODEL] at [LOCATION]
```

#### Roadside Assistance Templates

```
My [car/vehicle] [has a flat tire/won't start/battery is dead] at [LOCATION]
Need help with [flat tire/battery/locked out] at [LOCATION]
I'm stranded at [LOCATION], my [VEHICLE_MAKE] [VEHICLE_MODEL] [ISSUE]
Can someone help me with my [ISSUE] at [LOCATION]?
My [VEHICLE_MAKE]'s battery died at [LOCATION]
```

#### Appointment Templates

```
I want to schedule [an oil change/maintenance/service] for my [VEHICLE_MAKE] [VEHICLE_MODEL]
Can I book an appointment for [SERVICE_TYPE] on [DATE] at [TIME]?
Need to set up [SERVICE_TYPE] for next [week/Tuesday/DATE]
I'd like to bring my [VEHICLE_MAKE] in for [SERVICE_TYPE]
Is there availability for [SERVICE_TYPE] on [DATE]?
```

### 2. Entity Expansion Lists

Use these lists to fill in the template slots:

#### Vehicle Makes

```
Honda, Toyota, Ford, Chevrolet, Nissan, BMW, Mercedes, Audi, Hyundai, Kia, Subaru, Volkswagen, Lexus, Jeep, Dodge, Tesla, Mazda, Volvo, Acura, Infiniti
```

#### Vehicle Models

```
Civic, Accord, Camry, Corolla, F-150, Silverado, Altima, 3-Series, C-Class, A4, Elantra, Sonata, Soul, Optima, Forester, Outback, Jetta, Passat, RX, Grand Cherokee, Wrangler, Model 3, Model S, CX-5, S60, MDX, Q50
```

#### Locations

```
123 Main Street, downtown, the shopping mall, Walmart parking lot, near the gas station, 456 Oak Avenue, the highway, I-95, exit 23, the intersection of First and Main, my home, work, the office, [CITY_NAME] Plaza, north side of town
```

#### Service Types

```
oil change, tire rotation, brake service, inspection, engine diagnostic, tune-up, fluid check, air conditioning service, transmission service, wheel alignment, battery replacement, filter replacement
```

#### Issues

```
flat tire, dead battery, won't start, overheating, strange noise, check engine light, locked out, out of gas, smoking engine, brake problem, transmission problem
```

### 3. Variation Techniques

#### Adding Modifiers

Add words that don't change intent but add variety:

- Time urgency: "immediately", "as soon as possible", "urgently", "when possible"
- Politeness: "please", "if possible", "would appreciate"
- Context: "on my way to work", "during my vacation", "while traveling"

#### Grammatical Variations

- Questions vs. statements: "Can I get a tow?" vs. "I need a tow"
- Passive vs. active: "My car needs to be towed" vs. "I need a tow for my car"
- Formal vs. casual: "I would like to request" vs. "I need"

#### Handling Negation

Include examples with negation:

- "I don't need a tow anymore, just roadside assistance"
- "Not a flat tire, my battery is dead"

## Example Expansion Set

Here are 10 complete examples you can directly add to your training data:

```json
[
  {
    "text": "My Tesla Model 3 won't start near the mall on Oak Street",
    "intent": "roadside_request_battery",
    "entities": [
      { "entity": "vehicle_make", "value": "Tesla" },
      { "entity": "vehicle_model", "value": "Model 3" },
      { "entity": "pickup_location", "value": "near the mall on Oak Street" }
    ]
  },
  {
    "text": "Can I schedule an oil change for my Ford F-150 on Tuesday at 3pm?",
    "intent": "appointment_book_service",
    "entities": [
      { "entity": "service_type", "value": "oil change" },
      { "entity": "vehicle_make", "value": "Ford" },
      { "entity": "vehicle_model", "value": "F-150" },
      { "entity": "date", "value": "Tuesday" },
      { "entity": "time", "value": "3pm" }
    ]
  },
  {
    "text": "Need a tow ASAP at I-95 exit 43, my Jeep Wrangler broke down",
    "intent": "towing_request_tow",
    "entities": [
      { "entity": "pickup_location", "value": "I-95 exit 43" },
      { "entity": "vehicle_make", "value": "Jeep" },
      { "entity": "vehicle_model", "value": "Wrangler" }
    ]
  },
  {
    "text": "Locked myself out of my Honda Civic at the grocery store parking lot",
    "intent": "roadside_request_lockout",
    "entities": [
      { "entity": "vehicle_make", "value": "Honda" },
      { "entity": "vehicle_model", "value": "Civic" },
      { "entity": "pickup_location", "value": "grocery store parking lot" }
    ]
  },
  {
    "text": "Is there any availability for a brake inspection next Monday morning?",
    "intent": "appointment_availability_check",
    "entities": [
      { "entity": "service_type", "value": "brake inspection" },
      { "entity": "date", "value": "next Monday" },
      { "entity": "time", "value": "morning" }
    ]
  },
  {
    "text": "My tire is flat and I'm stranded on Highway 66 near the gas station",
    "intent": "roadside_request_tire",
    "entities": [
      { "entity": "issue", "value": "flat tire" },
      {
        "entity": "pickup_location",
        "value": "Highway 66 near the gas station"
      }
    ]
  },
  {
    "text": "Can you tow my BMW to the dealership on Main Street? I'm at work now",
    "intent": "towing_request_tow",
    "entities": [
      { "entity": "vehicle_make", "value": "BMW" },
      { "entity": "pickup_location", "value": "work" },
      { "entity": "destination", "value": "dealership on Main Street" }
    ]
  },
  {
    "text": "Need to reschedule my Toyota Camry's tune-up from Thursday to Friday afternoon",
    "intent": "appointment_reschedule",
    "entities": [
      { "entity": "vehicle_make", "value": "Toyota" },
      { "entity": "vehicle_model", "value": "Camry" },
      { "entity": "service_type", "value": "tune-up" },
      { "entity": "original_date", "value": "Thursday" },
      { "entity": "new_date", "value": "Friday" },
      { "entity": "time", "value": "afternoon" }
    ]
  },
  {
    "text": "My Nissan Altima is overheating on Broadway and 5th, need assistance please",
    "intent": "roadside_request_general",
    "entities": [
      { "entity": "vehicle_make", "value": "Nissan" },
      { "entity": "vehicle_model", "value": "Altima" },
      { "entity": "issue", "value": "overheating" },
      { "entity": "pickup_location", "value": "Broadway and 5th" }
    ]
  },
  {
    "text": "What's the estimated cost for transmission service on a 2018 Audi A4?",
    "intent": "appointment_service_cost_inquiry",
    "entities": [
      { "entity": "service_type", "value": "transmission service" },
      { "entity": "vehicle_year", "value": "2018" },
      { "entity": "vehicle_make", "value": "Audi" },
      { "entity": "vehicle_model", "value": "A4" }
    ]
  }
]
```

## Iterative Improvement Process

1. **Add Batch of Examples**: Add 10-20 new examples following the templates
2. **Retrain Model**: Run `python train.py`
3. **Test**: Run integration tests and manually test edge cases
4. **Analyze Errors**: Identify where the model fails
5. **Target Improvements**: Add more examples specifically for failure cases
6. **Repeat**: Continue this process until performance is satisfactory

## Additional Data Sources

Consider these additional sources for expanding your training data:

1. **Customer Service Logs**: Real conversations from support interactions
2. **User Testing**: Collect examples from user testing sessions
3. **Online Forums**: Gather real questions from automotive forums
4. **Data Augmentation Tools**: Use NLP libraries for automated data augmentation
5. **Synthetic Data Generation**: Use LLMs to generate realistic variations

## Balancing the Dataset

Ensure a balanced dataset by:

1. Counting examples per intent category
2. Aiming for at least 20-30 examples per intent
3. Adding more examples for intents with higher error rates
4. Including a diverse range of entities for each intent

By following these strategies, you can systematically improve your NLU model's performance across all intents and entity types.
