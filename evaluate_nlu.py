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

def track_metrics(metrics, output_dir="benchmark_results"):
    """
    Track metrics over time in a CSV file for historical comparison.
    Uses efficient file operations and minimal data duplication.

    Args:
        metrics (dict): Metrics dictionary from evaluate_model
        output_dir (str): Directory to save results

    Returns:
        pd.DataFrame: Historical metrics data
    """
    import csv
    from datetime import datetime
    import fcntl  # For file locking

    # Create metrics history file path
    history_file = os.path.join(output_dir, "metrics_history.csv")

    # Extract timestamp information
    timestamp = datetime.now().isoformat()
    date = datetime.now().strftime("%Y-%m-%d")

    # Extract only essential metrics to avoid bloat
    intent_metrics = metrics['intent_metrics']
    intent_accuracy = intent_metrics['accuracy']
    intent_precision = intent_metrics['precision']
    intent_recall = intent_metrics['recall']
    intent_f1 = intent_metrics['f1']

    # Get entity metrics efficiently with proper defaults
    entity_metrics = metrics.get('entity_metrics', {})
    entity_precision = entity_metrics.get('micro avg', {}).get('precision', 0.0)
    entity_recall = entity_metrics.get('micro avg', {}).get('recall', 0.0)
    entity_f1 = entity_metrics.get('micro avg', {}).get('f1-score', 0.0)

    # Use error analysis that's already computed instead of recalculating
    error_analysis = metrics.get('error_analysis', {})
    intent_error_rate = error_analysis.get('intent_error_rate', 0.0)
    entity_error_rate = error_analysis.get('entity_error_rate', 0.0)

    # Create a concise new row with just the needed fields
    new_row = {
        'timestamp': timestamp,
        'date': date,
        'intent_accuracy': intent_accuracy,
        'intent_precision': intent_precision,
        'intent_recall': intent_recall,
        'intent_f1': intent_f1,
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'entity_f1': entity_f1,
        'intent_error_rate': intent_error_rate,
        'entity_error_rate': entity_error_rate,
        # Add model identifier if available
        'model_id': metrics.get('model_id', 'unknown')
    }

    logger.info("Tracking metrics in history file")

    # Atomic file operations with proper locking to prevent corruption
    try:
        if os.path.exists(history_file):
            # Read existing data first
            history_df = pd.read_csv(history_file)

            # Append new row
            new_df = pd.DataFrame([new_row])
            history_df = pd.concat([history_df, new_df], ignore_index=True)
        else:
            # Create new DataFrame with single row
            history_df = pd.DataFrame([new_row])

        # Use file locking for atomic write
        with open(history_file, 'w') as f:
            try:
                # Get exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Write the updated DataFrame
                history_df.to_csv(f, index=False)

                # Release lock
                fcntl.flock(f, fcntl.LOCK_UN)
            except IOError:
                logger.warning("Could not immediately acquire lock for metrics history file - waiting...")
                # Wait for lock (blocking)
                fcntl.flock(f, fcntl.LOCK_EX)
                history_df.to_csv(f, index=False)
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Error updating metrics history: {str(e)}")
        # Still return history_df even if save failed

    logger.info(f"Metrics history updated in {history_file}")
    return history_df

def create_visualizations(metrics, history_df, output_dir="benchmark_results"):
    """
    Create optimized visualizations of benchmark results.
    Only generates relevant visualizations and uses efficient rendering.

    Args:
        metrics (dict): Metrics dictionary from evaluate_model
        history_df (pd.DataFrame): Historical metrics data
        output_dir (str): Directory to save visualizations

    Returns:
        dict: Paths to generated visualization files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    import hashlib

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory for visualizations if it doesn't exist
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Dictionary to track generated visualization files
    generated_files = {}

    # Only import matplotlib when needed
    plt.style.use('ggplot')  # Use a more modern style

    # Calculate hash of confusion matrix to detect changes
    confusion_matrix = np.array(metrics['intent_metrics']['confusion_matrix'])
    labels = metrics['intent_metrics']['labels']
    cm_hash = hashlib.md5(confusion_matrix.tobytes() + str(labels).encode()).hexdigest()

    # 1. Confusion matrix - only generate if it's interesting (more than one class)
    if len(labels) > 1:
        # Configure figure size based on matrix dimensions
        fig_size = min(12, max(8, len(labels) * 0.6))
        plt.figure(figsize=(fig_size, fig_size))

        # Normalize matrix for better visualization if values are large
        if np.sum(confusion_matrix) > 100:
            # Show percentages for large matrices
            sns.heatmap(
                confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True),
                annot=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                vmin=0, vmax=1
            )
            plt.title('Intent Classification Confusion Matrix (Normalized)')
        else:
            # Show raw counts for smaller matrices
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels
            )
            plt.title('Intent Classification Confusion Matrix')

        plt.xlabel('Predicted Intent')
        plt.ylabel('True Intent')
        plt.tight_layout()

        # Rotate labels if there are many classes
        if len(labels) > 10:
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)

        # Save with hash in filename for caching purposes
        cm_file = os.path.join(vis_dir, f"confusion_matrix_{cm_hash[:8]}.png")
        plt.savefig(cm_file, dpi=100)
        plt.close()

        generated_files['confusion_matrix'] = cm_file
        logger.info(f"Generated confusion matrix visualization: {cm_file}")

    # 2. Performance history - only if we have enough data points
    if len(history_df) > 1:
        # Calculate a hash of the history data
        history_hash = hashlib.md5(history_df.to_json().encode()).hexdigest()

        # Check if we have meaningful changes in the historical data
        # Only create this visualization if there are at least 2 data points or significant changes
        history_variance = history_df['intent_f1'].var()
        if len(history_df) >= 3 or history_variance > 0.0001:
            plt.figure(figsize=(12, 6))

            # Use efficient plotting with fewer individual plot calls
            ax = plt.subplot(1, 2, 1)
            # Combine metrics into a single DataFrame for more efficient plotting
            intent_metrics_df = history_df[['date', 'intent_accuracy', 'intent_precision', 'intent_recall', 'intent_f1']]
            intent_metrics_df = intent_metrics_df.set_index('date')
            intent_metrics_df.plot(ax=ax, marker='o')

            ax.set_title('Intent Classification Metrics Over Time')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            # Entity metrics in second subplot
            ax = plt.subplot(1, 2, 2)
            entity_metrics_df = history_df[['date', 'entity_precision', 'entity_recall', 'entity_f1']]
            entity_metrics_df = entity_metrics_df.set_index('date')
            entity_metrics_df.plot(ax=ax, marker='o')

            ax.set_title('Entity Recognition Metrics Over Time')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            plt.tight_layout()
            history_file = os.path.join(vis_dir, "performance_history.png")
            plt.savefig(history_file, dpi=100)
            plt.close()

            generated_files['performance_history'] = history_file
            logger.info(f"Generated performance history visualization: {history_file}")

            # Also generate a focused recent history if we have many data points
            if len(history_df) > 10:
                plt.figure(figsize=(10, 6))
                recent_df = history_df.iloc[-10:]

                # Plot only F1 scores for clarity in recent history
                recent_df.set_index('date')[['intent_f1', 'entity_f1']].plot(
                    marker='o',
                    figsize=(10, 6)
                )

                plt.title('Recent F1 Score Trends (Last 10 Runs)')
                plt.ylabel('F1 Score')
                plt.ylim(0, 1)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(['Intent F1', 'Entity F1'])
                plt.tight_layout()

                recent_file = os.path.join(vis_dir, "recent_performance.png")
                plt.savefig(recent_file, dpi=100)
                plt.close()

                generated_files['recent_performance'] = recent_file
                logger.info(f"Generated recent performance visualization: {recent_file}")

    # 3. Intent F1 scores by class - more intelligently rendered
    per_class_report = metrics['intent_metrics']['per_class_report']
    intents = list(per_class_report.keys())
    f1_scores = [per_class_report[intent]['f1-score'] for intent in intents]
    support = [per_class_report[intent]['support'] for intent in intents]

    # Skip this visualization if there are too few classes
    if len(intents) > 3:
        # Calculate dynamic figure size based on number of intents
        plt.figure(figsize=(10, max(6, min(20, len(intents) * 0.3))))

        # Sort by F1 score and limit to most interesting classes
        # Use numpy for more efficient sorting
        sorted_indices = np.argsort(f1_scores)

        # If there are many classes, focus on the worst performers
        if len(intents) > 20:
            # Show bottom 10 and top 5 performers for very large datasets
            display_indices = list(sorted_indices[:10]) + list(sorted_indices[-5:])
            labels = ["Bottom 10 and Top 5 classes by F1 Score"]
        else:
            # Otherwise show all classes
            display_indices = sorted_indices
            labels = ["All classes by F1 Score"]

        sorted_intents = [intents[i] for i in display_indices]
        sorted_f1 = [f1_scores[i] for i in display_indices]
        sorted_support = [support[i] for i in display_indices]

        # Create horizontal bar chart
        bars = plt.barh(sorted_intents, sorted_f1, color=plt.cm.viridis(np.array(sorted_f1)))

        # Add support count as text
        for i, (bar, sup) in enumerate(zip(bars, sorted_support)):
            plt.text(
                max(bar.get_width() + 0.02, 0.02),
                i,
                f'n={sup}',
                va='center',
                fontsize=8,
                alpha=0.7
            )

        plt.xlabel('F1 Score')
        plt.title('F1 Score by Intent Class')
        plt.xlim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        plt.tight_layout()

        intent_f1_file = os.path.join(vis_dir, f"intent_f1_scores_{timestamp}.png")
        plt.savefig(intent_f1_file, dpi=100)
        plt.close()

        generated_files['intent_f1_scores'] = intent_f1_file
        logger.info(f"Generated intent F1 scores visualization: {intent_f1_file}")

    # 4. Error distribution chart - new visualization showing error patterns
    if 'error_analysis' in metrics and len(metrics['error_analysis'].get('most_common_errors', [])) > 0:
        error_data = metrics['error_analysis']['most_common_errors']

        # Only create this chart if we have meaningful errors to show
        if len(error_data) >= 3:
            plt.figure(figsize=(10, 5))

            labels = [f"{true} → {pred}" for (true, pred), _ in error_data[:8]]
            values = [count for _, count in error_data[:8]]

            # Use horizontal bar chart for better label readability
            plt.barh(labels, values, color='salmon')
            plt.xlabel('Number of Occurrences')
            plt.title('Most Common Error Patterns')
            plt.tight_layout()

            errors_file = os.path.join(vis_dir, f"error_patterns_{timestamp}.png")
            plt.savefig(errors_file, dpi=100)
            plt.close()

            generated_files['error_patterns'] = errors_file
            logger.info(f"Generated error patterns visualization: {errors_file}")

    # Generate a summary visualization with all the key metrics
    plt.figure(figsize=(8, 6))

    metrics_summary = {
        'Intent Accuracy': metrics['intent_metrics']['accuracy'],
        'Intent F1': metrics['intent_metrics']['f1'],
        'Entity F1': metrics.get('entity_metrics', {}).get('micro avg', {}).get('f1-score', 0.0),
        'Intent Error Rate': metrics['error_analysis'].get('intent_error_rate', 0.0),
        'Entity Error Rate': metrics['error_analysis'].get('entity_error_rate', 0.0)
    }

    # Plot horizontal bars for key metrics
    plt.barh(
        list(metrics_summary.keys()),
        list(metrics_summary.values()),
        color=plt.cm.RdYlGn(np.array(list(metrics_summary.values())))
    )

    plt.xlim(0, 1)
    plt.xlabel('Score')
    plt.title('NLU Model Performance Summary')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')

    # Add value labels
    for i, (k, v) in enumerate(metrics_summary.items()):
        plt.text(max(v + 0.02, 0.02), i, f'{v:.4f}', va='center')

    plt.tight_layout()
    summary_file = os.path.join(vis_dir, f"metrics_summary_{timestamp}.png")
    plt.savefig(summary_file, dpi=100)
    plt.close()

    generated_files['metrics_summary'] = summary_file
    logger.info(f"Generated metrics summary visualization: {summary_file}")

    return generated_files

def generate_html_report(metrics, visualization_files, output_dir="benchmark_results"):
    """
    Generate a standalone HTML report that includes all metrics and visualizations.

    Args:
        metrics (dict): Metrics dictionary from evaluate_model
        visualization_files (dict): Dictionary of visualization file paths
        output_dir (str): Directory to save the report

    Returns:
        str: Path to the generated HTML report
    """
    from datetime import datetime
    import base64

    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"nlu_benchmark_report_{timestamp}.html")

    # Function to embed an image in HTML
    def embed_image(image_path):
        if not os.path.exists(image_path):
            return f"<p>Image not found: {image_path}</p>"

        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded_image}" style="max-width:100%">'

    # Extract key metrics
    intent_accuracy = metrics['intent_metrics']['accuracy']
    intent_f1 = metrics['intent_metrics']['f1']

    entity_metrics = metrics.get('entity_metrics', {})
    entity_f1 = entity_metrics.get('micro avg', {}).get('f1-score', 0)

    error_analysis = metrics.get('error_analysis', {})
    intent_error_rate = error_analysis.get('intent_error_rate', 0)
    entity_error_rate = error_analysis.get('entity_error_rate', 0)

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>

<html>
<head>
    <title>NLU Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .metric-card {{ background-color: #f1f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4a90e2; }}
        .metric {{ font-size: 18px; margin: 10px 0; }}
        .value {{ font-weight: bold; }}
        .row {{ display: flex; flex-wrap: wrap; margin: -10px; }}
        .col {{ flex: 1; padding: 10px; min-width: 300px; }}
        .viz-container {{ margin: 20px 0; background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .errors {{ background-color: #fff5f5; border-left: 4px solid #e74c3c; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
        .summary-section {{ border-top: 1px solid #eee; margin-top: 30px; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NLU Model Benchmark Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="metric-card">
            <h2>Performance Summary</h2>
            <div class="row">
                <div class="col">
                    <div class="metric">Intent Accuracy: <span class="value">{intent_accuracy:.4f}</span></div>
                    <div class="metric">Intent F1 Score: <span class="value">{intent_f1:.4f}</span></div>
                </div>
                <div class="col">
                    <div class="metric">Entity F1 Score: <span class="value">{entity_f1:.4f}</span></div>
                    <div class="metric">Intent Error Rate: <span class="value">{intent_error_rate:.4f}</span></div>
                </div>
            </div>
        </div>
        
        <h2>Visualizations</h2>
"""
    
    # Add each visualization if available
    if 'metrics_summary' in visualization_files:
        html_content += f"""
        <div class="viz-container">
            <h3>Metrics Summary</h3>
            {embed_image(visualization_files['metrics_summary'])}
        </div>
        """
    
    if 'confusion_matrix' in visualization_files:
        html_content += f"""
        <div class="viz-container">
            <h3>Intent Confusion Matrix</h3>
            {embed_image(visualization_files['confusion_matrix'])}
        </div>
        """
    
    if 'intent_f1_scores' in visualization_files:
        html_content += f"""
        <div class="viz-container">
            <h3>Intent F1 Scores by Class</h3>
            {embed_image(visualization_files['intent_f1_scores'])}
        </div>
        """
    
    if 'performance_history' in visualization_files:
        html_content += f"""
        <div class="viz-container">
            <h3>Performance History</h3>
            {embed_image(visualization_files['performance_history'])}
        </div>
        """
    
    if 'recent_performance' in visualization_files:
        html_content += f"""
        <div class="viz-container">
            <h3>Recent Performance Trends</h3>
            {embed_image(visualization_files['recent_performance'])}
        </div>
        """
    
    if 'error_patterns' in visualization_files:
        html_content += f"""
        <div class="viz-container errors">
            <h3>Error Analysis</h3>
            {embed_image(visualization_files['error_patterns'])}
        </div>
        """
    
    # Add entity performance table if available
    if 'micro avg' in entity_metrics:
        html_content += f"""
        <h2>Entity Recognition Performance</h2>
        <table>
            <tr>
                <th>Entity Type</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Support</th>
            </tr>
        """
        
        # Add rows for each entity type, but skip BIO tag prefixes
        for entity_type, scores in entity_metrics.items():
            if entity_type in ['micro avg', 'macro avg', 'weighted avg']:
                continue
            
            # Clean up entity type name (remove BIO prefixes)
            entity_name = entity_type
            if entity_type.startswith('B-') or entity_type.startswith('I-'):
                entity_name = entity_type[2:]
                
            # Skip if we've already added this entity type (avoid duplicates from B-/I- prefixes)
            if f'B-{entity_name}' in entity_metrics and entity_type != f'B-{entity_name}':
                continue
                
            html_content += f"""
            <tr>
                <td>{entity_name}</td>
                <td>{scores.get('precision', 0):.4f}</td>
                <td>{scores.get('recall', 0):.4f}</td>
                <td>{scores.get('f1-score', 0):.4f}</td>
                <td>{scores.get('support', 0)}</td>
            </tr>
            """
        
        # Add aggregate rows
        for agg_type in ['micro avg', 'macro avg', 'weighted avg']:
            if agg_type in entity_metrics:
                scores = entity_metrics[agg_type]
                html_content += f"""
                <tr>
                    <th>{agg_type}</th>
                    <td>{scores.get('precision', 0):.4f}</td>
                    <td>{scores.get('recall', 0):.4f}</td>
                    <td>{scores.get('f1-score', 0):.4f}</td>
                    <td>{scores.get('support', 0)}</td>
                </tr>
                """
        
        html_content += "</table>"
    
    # Add error examples if available
    if 'detailed_results' in metrics:
        intent_errors = [r for r in metrics['detailed_results'] if not r['intent_correct']]
        if intent_errors:
            html_content += f"""
            <h2>Error Examples</h2>
            <div class="metric-card errors">
                <h3>Intent Classification Errors ({len(intent_errors)} examples)</h3>
                <table>
                    <tr>
                        <th>Text</th>
                        <th>True Intent</th>
                        <th>Predicted Intent</th>
                        <th>Confidence</th>
                    </tr>
            """
            
            # Add rows for top errors (limited to avoid huge reports)
            for error in intent_errors[:20]:  # Limit to 20 examples
                html_content += f"""
                <tr>
                    <td>{error['text']}</td>
                    <td>{error['true_intent']}</td>
                    <td>{error['pred_intent']}</td>
                    <td>{error['confidence']:.4f}</td>
                </tr>
                """
                
            html_content += """
                </table>
            </div>
            """
    
    # Close the HTML document
    html_content += """
        <div class="summary-section">
            <h2>Next Steps</h2>
            <ul>
                <li>Review the error patterns to identify common misclassifications</li>
                <li>Consider adding more examples for intents with low F1 scores</li>
                <li>Check for entity recognition issues, especially for critical entity types</li>
                <li>Compare this report with previous benchmarks to track improvements</li>
            </ul>
        </div>
    </div>
</body>
</html>
    """
    
    # Write the HTML report to a file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report: {report_file}")
    return report_file

if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # For non-interactive environments
    import time

    # Start timing
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Evaluate NLU model performance')
    parser.add_argument('--benchmark', default='data/benchmark_dataset.json', 
                        help='Path to benchmark data (default: data/benchmark_dataset.json)')
    parser.add_argument('--model', default='trained_nlu_model', 
                        help='Path to trained model (default: trained_nlu_model)')
    parser.add_argument('--output', default='benchmark_results', 
                        help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--no-vis', action='store_true', 
                        help='Disable visualizations')
    parser.add_argument('--html-report', action='store_true', 
                        help='Generate HTML report')
    parser.add_argument('--model-id', 
                        help='Optional identifier for this model version')
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
        # Run evaluation
        metrics = evaluate_model(args.benchmark, args.model, args.output)

        # Update metrics with model_id if provided
        if args.model_id:
            metrics['model_id'] = args.model_id

        # Track metrics over time (always do this as it's lightweight)
        history_df = track_metrics(metrics, args.output)

        # Generate visualizations if not disabled (use caching for efficiency)
        visualization_files = {}
        if not args.no_vis:
            visualization_files = create_visualizations(metrics, history_df, args.output)

        # Generate HTML report if requested
        if args.html_report and visualization_files:
            report_file = generate_html_report(metrics, visualization_files, args.output)
            logger.info(f"HTML report generated: {report_file}")

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
            for i, ((true, pred), count) in enumerate(error_analysis['most_common_errors'][:5]):
                print(f"  {i+1}. {true} → {pred}: {count} occurrences")

        # Print time taken
        elapsed_time = time.time() - start_time
        print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")

        if not args.no_vis:
            print(f"\nVisualizations saved to: {os.path.join(args.output, 'visualizations')}")

        if args.html_report and 'report_file' in locals():
            print(f"HTML report: {report_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise 