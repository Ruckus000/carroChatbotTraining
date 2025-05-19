import json
import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from functools import lru_cache
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from seqeval.metrics import classification_report

# Import your NLU inferencer
from inference import NLUInferencer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nlu_eval')

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_benchmark_data(benchmark_data_path):
    """
    Load benchmark data from JSON file with error handling.

    Args:
        benchmark_data_path (str): Path to benchmark dataset

    Returns:
        list: Benchmark data examples
    """
    try:
        with open(benchmark_data_path, 'r') as f:
            benchmark_data = json.load(f)

        logger.info(f"Loaded {len(benchmark_data)} examples from benchmark dataset")
        return benchmark_data
    except FileNotFoundError:
        logger.error(f"Benchmark data file not found: {benchmark_data_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in benchmark data: {benchmark_data_path}")
        raise

def evaluate_model(benchmark_data_path, model_path="trained_nlu_model", output_dir="benchmark_results"):
    """
    Evaluate NLU model performance against a benchmark dataset.

    Args:
        benchmark_data_path (str): Path to the benchmark dataset JSON file
        model_path (str): Path to the trained model directory
        output_dir (str): Directory to save results

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load benchmark data
    benchmark_data = load_benchmark_data(benchmark_data_path)

    # Initialize the NLU model
    try:
        inferencer = NLUInferencer(model_path)
        logger.info(f"Initialized NLU model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize NLU model: {str(e)}")
        raise

    # Process examples and collect results
    results = process_benchmark_examples(benchmark_data, inferencer)

    # Calculate metrics from results
    metrics = calculate_metrics(results)

    # Save metrics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"metrics_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Evaluation complete. Results saved to {output_file}")
    return metrics

def process_benchmark_examples(benchmark_data, inferencer):
    """
    Process all benchmark examples and collect prediction results.

    Args:
        benchmark_data (list): List of benchmark examples
        inferencer (NLUInferencer): Initialized NLU model inferencer

    Returns:
        dict: Processed results including predictions and ground truth
    """
    # Prepare data structures
    true_intents = []
    pred_intents = []
    true_entities_tags = []
    pred_entities_tags = []
    detailed_results = []

    # Track error patterns during processing
    intent_errors = []
    entity_errors = []
    error_patterns = {}

    # Determine if we need entity evaluation (skip if no entities in benchmark)
    needs_entity_eval = any(len(example.get('entities', [])) > 0 for example in benchmark_data)

    # Process each example with progress bar
    for example in tqdm(benchmark_data, desc="Evaluating examples"):
        # Get true values
        true_intent = example['intent']
        true_entities = example.get('entities', [])

        # Get predictions
        try:
            prediction = inferencer.predict(example['text'])
            pred_intent = prediction['intent']['name']
            pred_entities = prediction.get('entities', [])
        except Exception as e:
            logger.warning(f"Prediction failed for: '{example['text']}' - {str(e)}")
            # Use fallback values for failed predictions
            pred_intent = "fallback_error"
            pred_entities = []

        # Add to intent evaluation lists
        true_intents.append(true_intent)
        pred_intents.append(pred_intent)

        # Only convert to BIO tags if entity evaluation is needed
        if needs_entity_eval:
            # Use the cached function
            true_bio = get_cached_bio_tags(example['text'], true_entities)
            pred_bio = get_cached_bio_tags(example['text'], pred_entities)

            true_entities_tags.append(true_bio)
            pred_entities_tags.append(pred_bio)

        # Create detailed result for this example
        entities_correct = are_entities_equal(true_entities, pred_entities)
        intent_correct = (true_intent == pred_intent)

        detailed_result = {
            'text': example['text'],
            'true_intent': true_intent,
            'pred_intent': pred_intent,
            'intent_correct': intent_correct,
            'true_entities': true_entities,
            'pred_entities': pred_entities,
            'entities_correct': entities_correct,
            'confidence': prediction['intent']['confidence']
        }

        detailed_results.append(detailed_result)

        # Collect error information inline
        if not intent_correct:
            intent_errors.append(detailed_result)
            pair = (true_intent, pred_intent)
            if pair not in error_patterns:
                error_patterns[pair] = []
            error_patterns[pair].append(example['text'])

        if not entities_correct:
            entity_errors.append(detailed_result)

    return {
        'true_intents': true_intents,
        'pred_intents': pred_intents,
        'true_entities_tags': true_entities_tags if needs_entity_eval else None,
        'pred_entities_tags': pred_entities_tags if needs_entity_eval else None,
        'detailed_results': detailed_results,
        'intent_errors': intent_errors,
        'entity_errors': entity_errors,
        'error_patterns': error_patterns,
        'needs_entity_eval': needs_entity_eval
    }

def calculate_metrics(results):
    """
    Calculate all evaluation metrics from processed results.

    Args:
        results (dict): Results from process_benchmark_examples

    Returns:
        dict: Complete metrics dictionary
    """
    true_intents = results['true_intents']
    pred_intents = results['pred_intents']
    detailed_results = results['detailed_results']

    # Calculate basic intent metrics
    intent_accuracy = accuracy_score(true_intents, pred_intents)
    intent_precision, intent_recall, intent_f1, _ = precision_recall_fscore_support(
        true_intents, pred_intents, average='weighted'
    )

    # Only calculate per-class metrics if we have multiple intent classes
    unique_intents = sorted(set(true_intents))
    if len(unique_intents) > 1:
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            true_intents, pred_intents, average=None, labels=unique_intents
        )

        per_class_report = {}
        for i, intent in enumerate(unique_intents):
            per_class_report[intent] = {
                'precision': per_class_precision[i],
                'recall': per_class_recall[i],
                'f1-score': per_class_f1[i],
                'support': per_class_support[i]
            }
    else:
        # Simplified metrics for single-class case
        per_class_report = {unique_intents[0]: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': len(true_intents)}}

    # Create confusion matrix
    cm = confusion_matrix(true_intents, pred_intents, labels=unique_intents)

    # Entity evaluation using seqeval (if entities present)
    if results['needs_entity_eval']:
        entity_report = classification_report(
            results['true_entities_tags'],
            results['pred_entities_tags'],
            output_dict=True
        )
    else:
        entity_report = {"note": "No entity evaluation performed - no entities in benchmark data"}

    # Calculate error metrics
    intent_errors = results['intent_errors']
    entity_errors = results['entity_errors']

    intent_error_rate = len(intent_errors) / len(detailed_results) if detailed_results else 0
    entity_error_rate = len(entity_errors) / len(detailed_results) if detailed_results else 0

    # Most common error patterns
    error_patterns = results['error_patterns']
    most_common_errors = sorted(
        [(k, len(v)) for k, v in error_patterns.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Confidence analysis for errors
    error_confidences = [r['confidence'] for r in intent_errors]
    avg_error_confidence = float(np.mean(error_confidences)) if error_confidences else 0

    # Compile all metrics
    metrics = {
        'intent_metrics': {
            'accuracy': intent_accuracy,
            'precision': intent_precision,
            'recall': intent_recall,
            'f1': intent_f1,
            'per_class_report': per_class_report,
            'confusion_matrix': cm.tolist(),
            'labels': unique_intents
        },
        'entity_metrics': entity_report,
        'error_analysis': {
            'intent_error_rate': intent_error_rate,
            'entity_error_rate': entity_error_rate,
            'most_common_errors': most_common_errors[:5],  # Just top 5 for efficiency
            'avg_error_confidence': avg_error_confidence
        },
        'detailed_results': detailed_results
    }

    return metrics

# Modified entity conversion function without using direct LRU cache on complex objects
def convert_entities_to_bio(text, entities):
    """
    Efficiently convert entity annotations to BIO tags for seqeval evaluation.

    Args:
        text (str): The original text
        entities (list): List of entity dictionaries

    Returns:
        list: List of BIO tags for each token in the text
    """
    # Tokenize the text (simple whitespace tokenization for now)
    tokens = text.split()

    # Initialize all tokens with 'O' tag
    bio_tags = ['O'] * len(tokens)

    # Skip processing if no entities
    if not entities:
        return bio_tags

    # Create a lower-case version of tokens for case-insensitive matching
    # (only do this once instead of repeatedly in the loop)
    tokens_lower = [t.lower().strip('.,!?;:') for t in tokens]

    for entity_obj in entities:
        entity_type = entity_obj.get('entity', '')
        entity_value = entity_obj.get('value', '')

        # Skip placeholders or empty values
        if not entity_value or (entity_value.startswith('[') and entity_value.endswith(']')):
            continue

        # Split and normalize entity value
        entity_tokens = entity_value.split()
        entity_tokens_lower = [t.lower().strip('.,!?;:') for t in entity_tokens]
        entity_len = len(entity_tokens)

        # Only search if entity can fit in text
        if entity_len <= len(tokens):
            # Use more efficient search
            for i in range(len(tokens) - entity_len + 1):
                # Quick length check before detailed comparison
                if all(tokens_lower[i+j] == entity_tokens_lower[j] for j in range(entity_len)):
                    # Found a match - tag with BIO
                    bio_tags[i] = f'B-{entity_type}'
                    for j in range(1, entity_len):
                        bio_tags[i+j] = f'I-{entity_type}'
                    break  # Assume first match is the correct one

    return bio_tags

# Use a simpler, more robust caching mechanism
_bio_cache = {}
def get_cached_bio_tags(text, entities):
    """
    Get BIO tags with a simpler caching mechanism.
    
    Args:
        text (str): The text to process
        entities (list): List of entity dictionaries
    
    Returns:
        list: List of BIO tags
    """
    # Create a cache key from text and a simplified representation of entities
    # Sort entities by entity type and value for consistent cache keys
    entities_key = []
    for e in sorted(entities, key=lambda x: (x.get('entity', ''), x.get('value', ''))):
        entity_type = e.get('entity', '')
        entity_value = e.get('value', '')
        entities_key.append((entity_type, entity_value))
    
    # Convert to a hashable tuple
    cache_key = (text, tuple(entities_key))
    
    # Check cache
    if cache_key in _bio_cache:
        return _bio_cache[cache_key]
    
    # Generate new tags
    bio_tags = convert_entities_to_bio(text, entities)
    
    # Cache the result (limit cache size)
    if len(_bio_cache) > 1000:  # Limit cache size
        _bio_cache.clear()
    _bio_cache[cache_key] = bio_tags
    
    return bio_tags

def are_entities_equal(true_entities, pred_entities):
    """
    Efficiently compare if two entity sets match exactly.

    Args:
        true_entities (list): Ground truth entities
        pred_entities (list): Predicted entities

    Returns:
        bool: True if the entity sets match
    """
    # Quick length check
    if len(true_entities) != len(pred_entities):
        return False

    # Early return for empty lists
    if not true_entities and not pred_entities:
        return True

    # Use sets for O(1) comparison instead of sorting and iterating
    true_norm = {(e['entity'], e['value'].lower()) for e in true_entities}
    pred_norm = {(e['entity'], e['value'].lower()) for e in pred_entities}

    return true_norm == pred_norm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate NLU model performance')
    parser.add_argument('--benchmark', default='data/benchmark_dataset.json', 
                        help='Path to benchmark data (default: data/benchmark_dataset.json)')
    parser.add_argument('--model', default='trained_nlu_model', 
                        help='Path to trained model (default: trained_nlu_model)')
    parser.add_argument('--output', default='benchmark_results', 
                        help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Automatically use nlu_benchmark_data.json if benchmark_dataset.json doesn't exist
    if args.benchmark == 'data/benchmark_dataset.json' and not os.path.exists(args.benchmark):
        alt_path = 'data/nlu_benchmark_data.json'
        if os.path.exists(alt_path):
            logger.info(f"Using alternative benchmark path: {alt_path}")
            args.benchmark = alt_path

    logger.info(f"Evaluating model against benchmark dataset: {args.benchmark}")

    try:
        metrics = evaluate_model(args.benchmark, args.model, args.output)

        # Print summary results
        print("\n===== SUMMARY RESULTS =====")
        print(f"Intent Accuracy: {metrics['intent_metrics']['accuracy']:.4f}")
        print(f"Intent F1 Score: {metrics['intent_metrics']['f1']:.4f}")

        # Entity results (if available)
        if 'micro avg' in metrics['entity_metrics']:
            entity_f1 = metrics['entity_metrics']['micro avg']['f1-score']
            print(f"Entity F1 Score: {entity_f1:.4f}")

        # Error analysis
        error_analysis = metrics['error_analysis']
        print(f"\nIntent Error Rate: {error_analysis['intent_error_rate']:.4f}")
        print(f"Entity Error Rate: {error_analysis['entity_error_rate']:.4f}")

        if error_analysis['most_common_errors']:
            print("\nMost common error patterns:")
            for (true, pred), count in error_analysis['most_common_errors']:
                print(f"  {true} → {pred}: {count} occurrences")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise 