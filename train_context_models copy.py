import os
import json
import logging
import random
import torch
import numpy as np
from transformers import (
    DistilBertTokenizer, 
    DistilBertConfig, 
    DistilBertForSequenceClassification, 
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set fixed seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Create output directories
def create_output_dirs(base_dir):
    """Create output directories for models and data."""
    os.makedirs(os.path.join(base_dir, "context_aware"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "context_aware", "negation_detector"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "context_aware", "context_switch_detector"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "context_aware"), exist_ok=True)

class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def generate_negation_dataset():
    """
    Generate a dataset for negation detection.
    
    Returns:
        Dictionary with texts and labels for negation detection
    """
    # Examples of negation phrases
    negation_examples = [
        "I don't need a tow truck",
        "I don't want roadside assistance",
        "I no longer need an appointment",
        "Actually, I don't need that",
        "I've changed my mind, I don't want that",
        "Forget about the tow truck",
        "Cancel my request",
        "I've decided against getting a tow",
        "I'm not interested in roadside assistance anymore",
        "I won't be needing that service",
        "Let's not proceed with the appointment",
        "I'd rather not have a tow truck",
        "I didn't say I needed a tow truck",
        "That's not what I'm looking for",
        "That's not what I meant",
        # Adding recommended additional negation examples
        "I no longer require that service",
        "Let's not do the tow truck after all",
        "I've decided I don't need assistance anymore",
        "On second thought, I don't want roadside help",
        "I'd rather not have a tow truck sent now"
    ]
    
    # Examples of non-negation phrases
    non_negation_examples = [
        "I need a tow truck",
        "I want roadside assistance",
        "I need to schedule an appointment",
        "Can you help me with my car",
        "My car broke down",
        "I have a flat tire",
        "My battery is dead",
        "I locked my keys in the car",
        "Can you send someone to help me",
        "I'd like to book a service appointment",
        "When can I bring my car in",
        "What services do you offer",
        "How much does towing cost",
        "Where can I get my car fixed",
        "Do you work on weekends",
        # Adding recommended contrastive examples
        "I need something different than a tow truck",
        "What I actually need is roadside assistance, not a tow",
        "Instead of towing, I need a battery jump"
    ]
    
    # Create variations by adding context and filler words
    contexts = ["", "Hi, ", "Hello, ", "Excuse me, ", "I was wondering, ", "Quick question, "]
    fillers = ["", " please", " right now", " as soon as possible", " thank you", " if that's possible"]
    
    # Generate variations
    varied_negation = []
    for example in negation_examples:
        varied_negation.append(example)
        for context in contexts:
            for filler in fillers:
                if context or filler:  # Only add if we're actually adding variation
                    varied_negation.append(f"{context}{example}{filler}")
    
    varied_non_negation = []
    for example in non_negation_examples:
        varied_non_negation.append(example)
        for context in contexts:
            for filler in fillers:
                if context or filler:  # Only add if we're actually adding variation
                    varied_non_negation.append(f"{context}{example}{filler}")
    
    # Create dataset
    texts = varied_negation + varied_non_negation
    labels = [1] * len(varied_negation) + [0] * len(varied_non_negation)
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return {
        "texts": texts,
        "labels": labels
    }

def generate_context_switch_dataset():
    """
    Generate a dataset for context switch detection.
    
    Returns:
        Dictionary with texts and labels for context switch detection
    """
    # Examples of context switch phrases
    context_switch_examples = {
        # General context switches (no specific services mentioned)
        "general": [
            "Actually, I need something else",
            "I changed my mind",
            "Forget what I said before",
            "On second thought",
            "I've reconsidered",
            "Let's do something different",
            "I'd prefer a different service",
            "I've changed my mind about what I need",
            "Actually, forget what I said before",
            "Scratch that, I need something else",
            "You know what, let's try something else",
            "I've decided to go a different route",
            "Let me change my request",
            "Nevermind",
            "I'm not interested in that",
            "My mistake, I need something else",
            "Wait, that's not right",
            "Hold on, that's not what I meant",
            "Let me start over",
            "Can we try a different approach?",
            "I think I misspoke earlier",
            "That's not exactly what I'm looking for",
            "Let's reset and try something else",
        ],
        
        # Specific service transitions (from towing to other services)
        "tow_to_roadside": [
            "Instead of a tow truck, I need a jump start",
            "I changed my mind, I need a battery jump instead of a tow",
            "Forget the tow truck, I need a tire change",
            "Rather than a tow, I need help with my keys",
            "On second thought, I need roadside assistance instead of towing",
            "I don't want a tow anymore, just help with my flat tire",
            "Instead of towing my car, can you just help me unlock it?",
            "I'd rather get a jump start than a tow",
            "My car doesn't need to be towed, it just needs fuel",
            "Actually, I just need someone to help me change my tire, not a full tow",
            "The tow won't be necessary, I just need help with my battery",
            "Let's skip the tow and just get someone to help with my keys",
            "I realized I don't need a tow, I just locked myself out",
            "Instead of a tow, can I get help with my overheated engine?",
        ],
        
        # Roadside to tow transitions
        "roadside_to_tow": [
            "This jump start isn't working, I think I need a tow instead",
            "Changing the tire won't fix it, I need a tow truck",
            "Instead of roadside assistance, I'll need a tow after all",
            "The battery is completely dead, I need a tow truck now",
            "This is worse than I thought, I need a full tow",
            "I don't think roadside help is enough, I need to get it towed",
            "The flat tire was just the start of the problem, I need a tow",
            "Actually, my car needs to be towed to a garage",
            "On second thought, roadside assistance won't fix this, I need a tow",
            "The problem is more serious than I realized, I need a tow truck",
        ],
        
        # Transitions to appointment scheduling
        "to_appointment": [
            "I've reconsidered, I'd like to schedule an appointment instead",
            "Instead of immediate help, I'd like to book a service for next week",
            "Let's schedule regular maintenance instead",
            "Actually, can we book a full service appointment?",
            "I'd rather make an appointment for an oil change",
            "Can we switch to scheduling a maintenance appointment?",
            "I think I should just make an appointment with a mechanic",
            "Let's forget the emergency service and schedule regular maintenance",
            "I'd prefer to book an appointment for tomorrow",
            "Instead of roadside help, I want to schedule a diagnostic appointment",
            "I changed my mind, I'd like to book a service appointment",
            "Rather than immediate help, I want to schedule a check-up for my car",
        ],
        
        # Transitions with timeframe changes
        "timeframe_changes": [
            "Actually, I don't need help right now, can we schedule for tomorrow?",
            "I changed my mind, I need this service ASAP, not next week",
            "Let's do this now instead of scheduling for later",
            "I need to postpone this request",
            "Can we move this up to today instead of next week?",
            "I'd rather deal with this later, not right now",
            "Actually, this is more urgent than I initially said",
            "Let's reschedule this for a different time",
            "I need to push this back a few days",
            "Can we make this more immediate?",
        ],
        
        # Transitions with location changes
        "location_changes": [
            "Actually, I'm not at the address I mentioned",
            "I changed my mind about the drop-off location",
            "Let's tow it to a different garage",
            "I need to update my pickup location",
            "Actually, I moved to a different spot",
            "I need to change where you're picking up my car",
            "The car isn't at the mall anymore, it's at my office",
            "I need to change the destination for the tow",
            "Let's reroute to a different service center",
            "I'm actually at a different location now",
        ],
        
        # Vehicle information changes
        "vehicle_changes": [
            "Actually, it's not a Honda, it's a Toyota",
            "I gave you the wrong car model, it's actually a Civic not an Accord",
            "The car is blue, not red like I said",
            "I mentioned the wrong year, it's a 2019 not a 2018",
            "It's actually my wife's car, not mine, so it's a different model",
            "The license plate I gave was incorrect",
            "It's a sedan, not an SUV like I mentioned",
            "The car details I gave were for my other vehicle",
            "I need to correct the vehicle information I provided",
            "The VIN I gave you was for my old car",
        ],
        
        # Service detail changes
        "service_detail_changes": [
            "I need a different kind of tire than I mentioned",
            "The battery type I need is different from what I said",
            "I need a more specialized service than I initially requested",
            "The problem isn't what I described earlier",
            "I misdiagnosed the issue with my car",
            "The service I requested won't fix my actual problem",
            "I need additional services beyond what I mentioned",
            "The issue is more complex than I initially described",
            "The oil type needs to be different from what I specified",
            "I forgot to mention some important details about the service I need",
        ]
    }
    
    # Examples of non-context switch phrases
    non_context_switch_examples = {
        # Basic towing requests
        "towing_basic": [
            "I need a tow truck",
            "Can I get a tow?",
            "My car needs to be towed",
            "I need towing service please",
            "Can you send a tow truck to my location?",
            "I'd like to request a tow truck",
            "My vehicle requires towing",
            "I need to get my car towed to a garage",
            "Can you arrange a tow for me?",
            "How soon can a tow truck get here?",
            "Is it possible to get a tow right now?",
            "Can someone tow my car from this location?",
            "I'm stranded and need a tow truck",
            "My car won't start, I need a tow",
            "Looking for towing assistance"
        ],
        
        # Detailed towing requests with specifics
        "towing_detailed": [
            "I need my Honda Civic towed from the mall",
            "Can I get a tow truck for a midsize SUV?",
            "My truck broke down on Highway 101, need a tow",
            "I need a flatbed tow truck specifically",
            "Can you tow my car from my home to the dealership?",
            "I need a tow truck that can handle an AWD vehicle",
            "My car has a transmission issue and needs towing",
            "Looking for a tow truck that can handle a large pickup",
            "Need to tow my sedan about 20 miles to the repair shop",
            "I have a compact car that needs to be towed",
            "My vehicle is in a tight parking spot and needs towing",
            "Can you tow my car from downtown to the north side?",
            "Need towing from the grocery store parking lot"
        ],
        
        # Basic roadside assistance requests
        "roadside_basic": [
            "I need roadside assistance",
            "Can someone help with my car?",
            "I need help with my vehicle",
            "I require roadside help",
            "Can you send roadside assistance please?",
            "I'm having car trouble and need help",
            "My vehicle is having issues",
            "I need someone to come look at my car",
            "Can you dispatch roadside help?",
            "Is roadside assistance available now?",
            "Can I get help with my car immediately?",
            "I'm stuck and need roadside assistance",
            "Need emergency assistance with my vehicle",
            "Looking for someone to help with car problems",
            "Can you send help for my vehicle?"
        ],
        
        # Specific roadside issues
        "roadside_specific": [
            "I have a flat tire",
            "My battery is dead",
            "I locked my keys in the car",
            "I ran out of gas",
            "My car is overheating",
            "The engine won't start",
            "My car won't go into gear",
            "I think my alternator died",
            "Need help changing a tire",
            "Could use a jump start for my battery",
            "I need help getting into my locked car",
            "My car stalled and won't restart",
            "The engine is making strange noises",
            "My brake pedal feels spongy",
            "My car shut off while driving and won't start again",
            "I need fuel delivery service",
            "Car won't start in cold weather",
            "Need help with jump starting my vehicle"
        ],
        
        # Appointment scheduling requests
        "appointment_basic": [
            "I need to schedule an appointment",
            "Can I book a service appointment?",
            "I'd like to make an appointment for car service",
            "When can I bring my car in?",
            "I need to set up a time for car maintenance",
            "Can I schedule my vehicle for service?",
            "I want to book an appointment for next week",
            "How do I make a service appointment?",
            "Looking to schedule my car for maintenance",
            "I need to arrange a time for car service",
            "Can I reserve a service slot for my vehicle?",
            "I'd like to set up regular maintenance",
            "Need to schedule my car for a check-up",
            "Can I make an appointment for automotive service?",
            "I want to bring my car in for service"
        ],
        
        # Specific service appointments
        "appointment_specific": [
            "I need to schedule an oil change",
            "Can I make an appointment for brake service?",
            "I'd like to book a tire rotation",
            "When can I bring my car in for A/C repair?",
            "Need to schedule transmission service",
            "I'd like to book a full vehicle inspection",
            "Can I schedule a battery replacement?",
            "I need to make an appointment for alignment service",
            "Looking to schedule a timing belt replacement",
            "Need to book my 30,000-mile service",
            "I'd like to schedule a diagnostic appointment",
            "Can I make an appointment for a tune-up?",
            "Need to schedule radiator service",
            "I'd like to book my car for filter replacements",
            "When can I bring my vehicle in for sensor replacement?"
        ],
        
        # Information requests
        "information_queries": [
            "What services do you offer?",
            "How much does towing cost?",
            "Where can I get my car fixed?",
            "Do you work on weekends?",
            "What are your business hours?",
            "Do you service electric vehicles?",
            "What's your service area?",
            "How long does an oil change typically take?",
            "Do you offer warranty work?",
            "What types of payment do you accept?",
            "Are there any current service promotions?",
            "Do I need an appointment for an oil change?",
            "Can you service imported vehicles?",
            "What's the average wait time for service?",
            "Do you provide loaner vehicles during service?"
        ],
        
        # Status inquiries
        "status_inquiries": [
            "When will the tow truck arrive?",
            "How long until help gets here?",
            "What's the ETA on roadside assistance?",
            "Is my appointment still scheduled for today?",
            "Has my service been completed yet?",
            "Is my car ready for pickup?",
            "Where is the tow truck now?",
            "How much longer until someone arrives?",
            "Has my appointment time been confirmed?",
            "Did you receive my service request?",
            "Can you check the status of my roadside assistance?",
            "Is there an update on when the technician will arrive?",
            "Has my service request been processed?",
            "What position am I in the service queue?",
            "Is my service still expected to be completed today?"
        ],
        
        # Location and vehicle details
        "location_vehicle_details": [
            "I'm at the corner of Main and 5th Street",
            "My car is a 2018 Toyota Camry",
            "I'm in the northern parking lot of the mall",
            "The vehicle is a silver Honda Civic",
            "I'm about 2 miles east of the highway exit",
            "It's a black Ford F-150 pickup",
            "I'm at the gas station on Highway 9",
            "My car is the blue sedan in front of the restaurant",
            "The vehicle is parked at 123 Oak Street",
            "It's a 2020 Chevrolet Equinox, red color",
            "I'm in the downtown area near the library",
            "My SUV is in the hospital parking garage",
            "The car is a white Nissan Altima",
            "I'm at the shopping center on Wilson Boulevard",
            "It's a gray Hyundai Sonata with tinted windows"
        ],
        
        # Follow-up questions from the user
        "user_followups": [
            "How much will this cost?",
            "Will insurance cover this?",
            "How long will this take?",
            "Do I need to be present?",
            "Can I pay with a credit card?",
            "Should I stay with my vehicle?",
            "Can I track the service vehicle?",
            "Will you call before arriving?",
            "Is there a membership discount?",
            "Do you need my VIN number?",
            "What information do you need from me?",
            "Is there a warranty on the service?",
            "Should I try to start it again?",
            "Do you need my exact location?",
            "Can the driver call me when they're close?"
        ]
    }
    
    # Create variations by adding context and filler words
    contexts = ["", "Hi, ", "Hello, ", "Excuse me, ", "I was wondering, ", "Quick question, "]
    fillers = ["", " please", " right now", " as soon as possible", " thank you", " if that's possible"]
    
    # Generate variations
    varied_context_switch = []
    for example in context_switch_examples:
        varied_context_switch.append(example)
        for context in contexts:
            for filler in fillers:
                if context or filler:  # Only add if we're actually adding variation
                    varied_context_switch.append(f"{context}{example}{filler}")
    
    varied_non_context_switch = []
    for category in non_context_switch_examples:
        for example in non_context_switch_examples[category]:
            varied_non_context_switch.append(example)
            for context in contexts:
                for filler in fillers:
                    if context or filler:  # Only add if we're actually adding variation
                        varied_non_context_switch.append(f"{context}{example}{filler}")
    
    # Create dataset
    texts = varied_context_switch + varied_non_context_switch
    labels = [1] * len(varied_context_switch) + [0] * len(varied_non_context_switch)
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return {
        "texts": texts,
        "labels": labels
    }

def save_model_with_config(model, tokenizer, output_dir, task_name):
    """
    Save model, tokenizer, and configuration in a format compatible with local loading.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        output_dir: Directory to save the model
        task_name: Name of the task (e.g., "negation_detector")
    """
    model_dir = os.path.join(output_dir, task_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model state and config
    model.save_pretrained(
        model_dir,
        save_config=True,
        save_function=torch.save,
        push_to_hub=False
    )
    
    # Save tokenizer files
    tokenizer.save_pretrained(
        model_dir,
        push_to_hub=False
    )
    
    # Save additional metadata
    with open(os.path.join(model_dir, "model_info.json"), "w") as f:
        json.dump({
            "model_type": "distilbert",
            "is_local": True,
            "task": task_name,
            "version": "1.0"
        }, f, indent=2)
    
    logger.info(f"Model and associated files saved to {model_dir}")
    return model_dir

def train_binary_classifier(task_name, texts, labels, output_dir):
    """
    Train a binary classifier for context-related tasks.
    
    Args:
        task_name: Name of the task (e.g., "negation", "context_switch")
        texts: List of text examples
        labels: Corresponding labels (0 or 1)
        output_dir: Directory to save the model
    
    Returns:
        Path to the trained model
    """
    logger.info(f"Training {task_name} classifier")
    
    # Split data into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Tokenize data
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)
    
    # Set up model configuration
    config = DistilBertConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        no_cuda=True
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # Save the model using the new function
    model_path = save_model_with_config(model, tokenizer, output_dir, task_name)
    
    # Save evaluation results
    with open(os.path.join(model_path, "eval_results.json"), "w") as f:
        json.dump(eval_result, f, indent=2)
    
    return model_path

def train_context_aware_models(output_dir="./output/models"):
    """
    Train all context-aware models.
    
    Args:
        output_dir: Directory to save the models
    """
    logger.info("Starting context-aware model training")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directories
    create_output_dirs(output_dir)
    
    # Generate negation dataset
    logger.info("Generating negation dataset")
    negation_data = generate_negation_dataset()
    
    # Generate context switch dataset
    logger.info("Generating context switch dataset")
    context_switch_data = generate_context_switch_dataset()
    
    # Save datasets
    data_dir = os.path.join(output_dir, "..", "data", "context_aware")
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "negation_dataset.json"), "w") as f:
        json.dump({
            "texts": list(negation_data["texts"]),
            "labels": list(negation_data["labels"])
        }, f)
    
    with open(os.path.join(data_dir, "context_switch_dataset.json"), "w") as f:
        json.dump({
            "texts": list(context_switch_data["texts"]),
            "labels": list(context_switch_data["labels"])
        }, f)
    
    # Train negation detector
    logger.info("Training negation detector")
    negation_model_path = train_binary_classifier(
        "negation_detector",
        negation_data["texts"],
        negation_data["labels"],
        os.path.join(output_dir, "context_aware")
    )
    
    # Train context switch detector
    logger.info("Training context switch detector")
    context_switch_model_path = train_binary_classifier(
        "context_switch_detector",
        context_switch_data["texts"],
        context_switch_data["labels"],
        os.path.join(output_dir, "context_aware")
    )
    
    logger.info("Context-aware model training complete")
    logger.info(f"Negation detector saved to {negation_model_path}")
    logger.info(f"Context switch detector saved to {context_switch_model_path}")
    
    return {
        "negation_detector": negation_model_path,
        "context_switch_detector": context_switch_model_path
    }

if __name__ == "__main__":
    train_context_aware_models() 