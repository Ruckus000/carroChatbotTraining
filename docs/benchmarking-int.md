Comprehensive NLU Model Benchmarking Implementation Plan
This plan outlines a multi-phase approach for implementing a robust NLU model performance benchmarking system with Streamlit visualization. Each phase builds upon the previous one and ends with validation steps to ensure proper implementation.

Phase 1: Core Evaluation Script Implementation
Step 1.1: Create Modular Evaluation Script Structure
Create a new file evaluate_nlu.py with optimized, modular evaluation logic:

python
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
        json.dump(metrics, f, indent=2)

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
            true_bio = convert_entities_to_bio(example['text'], true_entities)
            pred_bio = convert_entities_to_bio(example['text'], pred_entities)

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

@lru_cache(maxsize=128)
def convert_entities_to_bio(text, entities):
"""
Efficiently convert entity annotations to BIO tags for seqeval evaluation.
Uses caching to avoid redundant conversions.

    Args:
        text (str): The original text
        entities (tuple): Tuple of entity dictionaries (converted from list for caching)

    Returns:
        list: List of BIO tags for each token in the text
    """
    # Handle case where entities is a list (convert to tuple for caching)
    if isinstance(entities, list):
        # Convert to a tuple of frozensets for hashability
        entity_tuples = tuple(frozenset(e.items()) for e in entities)
        return convert_entities_to_bio(text, entity_tuples)

    # Tokenize the text (simple whitespace tokenization for now)
    tokens = text.split()

    # Initialize all tokens with 'O' tag
    bio_tags = ['O'] * len(tokens)

    # Convert tuple of frozensets back to list of dicts
    entities_list = [dict(e) for e in entities]

    # Skip processing if no entities
    if not entities_list:
        return bio_tags

    # Create a lower-case version of tokens for case-insensitive matching
    # (only do this once instead of repeatedly in the loop)
    tokens_lower = [t.lower().strip('.,!?;:') for t in tokens]

    for entity_obj in entities_list:
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

if **name** == "**main**":
import argparse

    parser = argparse.ArgumentParser(description='Evaluate NLU model performance')
    parser.add_argument('--benchmark', default='data/nlu_benchmark_data.json', help='Path to benchmark data')
    parser.add_argument('--model', default='trained_nlu_model', help='Path to trained model')
    parser.add_argument('--output', default='benchmark_results', help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

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

Step 1.2: Optimized Entity Handling
The entity handling is now optimized in the above code with:

Cache-enabled BIO tag conversion: Using lru_cache to avoid redundant conversions
Efficient entity comparison: Using sets for O(1) comparison
Skipping unnecessary conversions: Only performing entity evaluation if entities exist
Early returns: Quickly handling edge cases for better performance
Preprocessing text once: Converting tokens to lowercase once instead of repeatedly
Step 1.3: Modular Design with Better Error Handling
The code is now structured in a modular way:

Separate loading function: load_benchmark_data() for better error handling
Processing pipeline: process_benchmark_examples() to handle the evaluation loop
Metrics calculation: calculate_metrics() to generate all metrics
Integrated error analysis: Error collection during main processing instead of as a separate step
Proper logging: Using Python's logging module instead of print statements
Progress tracking: Adding tqdm progress bar for visibility during long runs
Try/except blocks: Gracefully handling failures in predictions or file operations
Step 1.4: Efficiency Improvements
Several efficiency improvements have been implemented:

Conditional metrics calculation: Only calculating per-class metrics for multiple classes
Optimized entity processing: More efficient entity matching algorithm
Caching for repeated operations: Using lru_cache for expensive operations
Low overhead error tracking: Integrated with main processing
Selective entity evaluation: Skip entity evaluation if no entities in benchmark data
Limiting error patterns: Only storing top 5 most common errors to save space
Phase 1 Validation Steps:
Run python evaluate_nlu.py to execute the evaluation
Check the logging output for proper initialization and progress tracking
Verify efficient handling of benchmark examples with the progress bar
Confirm that entity evaluation is skipped if no entities are present
Test error handling by intentionally using an invalid path
Check that metrics calculations are accurate and complete
Verify that the output JSON file contains all necessary information
Monitor memory usage to confirm efficiency improvements
Phase 2: Metrics Tracking and Efficient Visualization
Step 2.1: Implement Optimized Metrics Tracking
Create an efficient metrics tracking function in evaluate_nlu.py:

python
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

Step 2.2: Create Memory-Efficient Visualization Functions
Implement efficient visualization functions with intelligent caching and selective rendering:

python
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
        'Entity F1': entity_metrics.get('micro avg', {}).get('f1-score', 0.0),
        'Intent Error Rate': error_analysis.get('intent_error_rate', 0.0),
        'Entity Error Rate': error_analysis.get('entity_error_rate', 0.0)
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

Step 2.3: Create an HTML Report Generator (Optional but Valuable)
Add a function to generate a standalone HTML report that combines all visualizations:

python
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
Step 2.4: Update Main Function with Efficient Processing
Update the main function to use the new tracking and visualization features with intelligent defaults:

python
if **name** == "**main**":
import argparse
import matplotlib
matplotlib.use('Agg') # For non-interactive environments
import time

    # Start timing
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Evaluate NLU model performance')
    parser.add_argument('--benchmark', default='data/nlu_benchmark_data.json', help='Path to benchmark data')
    parser.add_argument('--model', default='trained_nlu_model', help='Path to trained model')
    parser.add_argument('--output', default='benchmark_results', help='Output directory for results')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--model-id', help='Optional identifier for this model version')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Evaluating model against benchmark dataset: {args.benchmark}")

    try:
        # Add model ID to metrics if provided
        kwargs = {}
        if args.model_id:
            kwargs['model_id'] = args.model_id

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

        # Entity results
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

Phase 2 Validation Steps:
Run python evaluate_nlu.py to generate metrics, tracking data, and visualizations
Verify that the metrics_history.csv file is created/updated correctly with efficient file operations
Check that visualizations are only generated when necessary (using caching where possible)
Run with different options to test the conditional behavior:
python evaluate_nlu.py --no-vis (skip visualizations)
python evaluate_nlu.py --html-report (generate HTML report)
python evaluate_nlu.py --model-id test_model_v1 (track model version)
Check memory usage during visualization to confirm efficiency
Test with large datasets to verify scaling behavior
Confirm that the HTML report provides comprehensive insights
Phase 3: Streamlit Dashboard Implementation
Step 3.1: Create an Efficient Streamlit Dashboard
Create a new file nlu_dashboard.py that implements a clean, efficient Streamlit interface:

python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
import time
from datetime import datetime
from functools import lru_cache

# Set page config for better UI

st.set_page_config(
page_title="NLU Model Benchmarking",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded"
)

# Apply clean styling

st.markdown("""

<style>
    .main {
        padding: 1rem 1.5rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 16px;
        line-height: 1.5;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 600;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 400;
        color: #555;
        text-transform: uppercase;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4b9dff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .card.warning {
        border-left: 4px solid #ff9d4b;
    }
    .card.success {
        border-left: 4px solid #4bff9d;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .header-title {
        font-size: 24px;
        font-weight: 600;
        margin: 0;
    }
    .header-subtitle {
        font-size: 16px;
        color: #555;
        margin: 0;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4b9dff;
    }
</style>

""", unsafe_allow_html=True)

# Constants

BENCHMARK_DIR = "benchmark_results"
HISTORY_FILE = os.path.join(BENCHMARK_DIR, "metrics_history.csv")

# Caching for expensive operations

@st.cache*data(ttl=60) # Cache for 1 minute
def load_available_runs():
"""Load available benchmark runs with efficient caching""" # Find all metrics files
metric_files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "metrics*\*.json")), reverse=True)
runs = []

    for file in metric_files:
        # Extract timestamp from filename
        timestamp = os.path.basename(file).replace("metrics_", "").replace(".json", "")
        try:
            # Format timestamp for display
            formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

            # Get file size for information
            file_size = os.path.getsize(file) / 1024  # KB

            # Get model_id if available (quick peek without loading full file)
            model_id = "Unknown"
            try:
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        if '"model_id":' in line:
                            model_id = line.split('"model_id":')[1].strip().strip('",')
                            break
                        if i > 10:  # Only check first few lines
                            break
            except:
                pass

            runs.append({
                "file": file,
                "timestamp": formatted_time,
                "raw_timestamp": timestamp,
                "model_id": model_id,
                "size": f"{file_size:.1f} KB"
            })
        except Exception as e:
            # Skip files with invalid naming
            continue

    return runs

@st.cache_data
def load_metrics(file_path):
"""Load metrics file with caching"""
try:
with open(file_path, 'r') as f:
return json.load(f)
except Exception as e:
st.error(f"Error loading metrics file: {str(e)}")
return None

@st.cache_data
def load_history():
"""Load metrics history with caching"""
if os.path.exists(HISTORY_FILE):
try:
return pd.read_csv(HISTORY_FILE)
except Exception as e:
st.error(f"Error loading history file: {str(e)}")
return None

def custom_metric_card(label, value, delta=None, help_text=None, color="blue"):
"""Create a custom, visually appealing metric card""" # Determine color based on value for visual feedback
if color == "auto":
if value > 0.9:
color = "green"
elif value > 0.7:
color = "blue"
elif value > 0.5:
color = "orange"
else:
color = "red"

    # Set color code
    color_code = {
        "blue": "#4b9dff",
        "green": "#4bff9d",
        "orange": "#ff9d4b",
        "red": "#ff4b4b"
    }.get(color, "#4b9dff")

    # Format value
    if isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)

    # Create HTML for metric card
    html = f"""
    <div style="border-left: 4px solid {color_code}; padding: 10px 15px; border-radius: 5px; background: #f8f9fa; margin-bottom: 10px;">
        <div style="font-size: 14px; color: #555; font-weight: 500;">{label}</div>
        <div style="font-size: 28px; font-weight: 600; margin: 5px 0;">{formatted_value}</div>
    """

    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_symbol = "▲" if delta >= 0 else "▼"
        html += f'<div style="font-size: 14px; color: {"#4bff9d" if delta_color == "green" else "#ff4b4b"};">{delta_symbol} {abs(delta):.4f}</div>'

    if help_text:
        html += f'<div style="font-size: 12px; color: #777; margin-top: 5px;">{help_text}</div>'

    html += "</div>"

    return html

def render_header():
"""Render dashboard header with information and controls"""
col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="header-container">'
                  '<div>'
                  '<h1 class="header-title">NLU Model Benchmarking Dashboard</h1>'
                  '<p class="header-subtitle">Track and analyze model performance metrics</p>'
                  '</div>'
                  '</div>', unsafe_allow_html=True)

    with col2:
        # Check when dashboard was last refreshed
        last_refresh = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"<div style='text-align: right; color: #777;'>Last refreshed: {last_refresh}</div>", unsafe_allow_html=True)
        # Add a refresh button
        if st.button("🔄 Refresh Data"):
            # Clear all cached data
            st.cache_data.clear()
            st.rerun()

def format_class_name(name, max_length=30):
"""Format class names for better display"""
if len(name) <= max_length:
return name

    # Determine where to truncate
    half = max_length // 2 - 2
    return f"{name[:half]}...{name[-half:]}"

def render_confusion_matrix(cm, labels, width=None, height=None):
"""Render a clean, readable confusion matrix""" # Scale figure size based on number of classes
n_classes = len(labels)

    if width is None:
        width = min(12, max(8, n_classes * 0.5))
    if height is None:
        height = min(12, max(8, n_classes * 0.5))

    # Format labels for better display
    display_labels = [format_class_name(label) for label in labels]

    fig, ax = plt.subplots(figsize=(width, height))

    # Normalize if many samples
    if np.sum(cm) > 100:
        im = sns.heatmap(
            cm.astype(float) / cm.sum(axis=1, keepdims=True),
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax
        )
        ax.set_title('Normalized Confusion Matrix')
    else:
        im = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax
        )
        ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    # Rotate labels if there are many classes
    if n_classes > 10:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    else:
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

    plt.tight_layout()
    return fig

def render_class_performance(class_report, class_type="intent"):
"""Render class performance visualization"""
if not class_report:
return None

    # Extract data from report
    classes = list(class_report.keys())
    f1_scores = [class_report[cls]['f1-score'] for cls in classes]
    support = [class_report[cls]['support'] for cls in classes]

    # Skip classes with specific names if needed
    if class_type == "entity":
        classes = [c for c in classes if not c.startswith('micro') and not c.startswith('macro')]
        if not classes:
            return None

    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)

    # Find low-performing classes to highlight
    low_performing_indices = [i for i in sorted_indices if f1_scores[i] < 0.7][:5]

    # If too many classes, focus on worst performers
    if len(classes) > 25:
        # Show bottom 10 and top 5 classes
        indices_to_show = list(sorted_indices[:10])
        # Add a few top performers
        indices_to_show.extend(sorted_indices[-5:])
        # Sort indices to maintain proper order
        indices_to_show.sort()
    else:
        indices_to_show = sorted_indices

    selected_classes = [classes[i] for i in indices_to_show]
    selected_f1 = [f1_scores[i] for i in indices_to_show]
    selected_support = [support[i] for i in indices_to_show]

    # Format class names for better display
    display_classes = [format_class_name(cls) for cls in selected_classes]

    # Create figure with adjusted size based on number of classes
    height = max(5, min(20, len(selected_classes) * 0.4))
    fig, ax = plt.subplots(figsize=(10, height))

    # Create color gradient based on F1 scores
    colors = plt.cm.RdYlGn(np.array(selected_f1))

    bars = ax.barh(display_classes, selected_f1, color=colors)

    # Add support count annotations
    for i, (bar, sup) in enumerate(zip(bars, selected_support)):
        ax.text(
            bar.get_width() + 0.02,
            i,
            f'n={sup}',
            va='center',
            alpha=0.7
        )

    # Highlight low-performing classes
    if low_performing_indices:
        if len(class_report) > 25:
            ax.text(0.5, 1.02, "⚠️ Consider improving low F1 score classes", transform=ax.transAxes,
                   ha='center', fontsize=12, color='#ff9d4b')

    ax.set_xlabel('F1 Score')
    ax.set_title(f'{class_type.capitalize()} Class F1 Scores')
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')

    plt.tight_layout()
    return fig

def main():
"""Main dashboard function"""
render_header()

    # Load available runs
    runs = load_available_runs()

    if not runs:
        st.warning("No benchmark results found. Run the evaluation script first.")
        st.markdown("""
        ### Getting Started
        To generate benchmark results:
        ```bash
        python evaluate_nlu.py --benchmark data/nlu_benchmark_data.json
        ```

        Once you've run the benchmark, refresh this page to view the results.
        """)
        return

    # Sidebar for run selection and comparison
    with st.sidebar:
        st.subheader("Benchmark Runs")

        # Run selection
        selected_run_index = st.selectbox(
            "Select a benchmark run:",
            range(len(runs)),
            format_func=lambda i: f"{runs[i]['timestamp']} ({runs[i]['model_id']})"
        )

        selected_file = runs[selected_run_index]["file"]
        metrics = load_metrics(selected_file)

        # Comparison selection (only if multiple runs available)
        compare_enabled = st.checkbox("Compare with another run", value=False, disabled=len(runs) < 2)

        if compare_enabled and len(runs) > 1:
            compare_options = [i for i in range(len(runs)) if i != selected_run_index]
            compare_run_index = st.selectbox(
                "Select run to compare:",
                compare_options,
                format_func=lambda i: f"{runs[i]['timestamp']} ({runs[i]['model_id']})"
            )
            compare_file = runs[compare_run_index]["file"]
            compare_metrics = load_metrics(compare_file)
        else:
            compare_metrics = None

        # Run details section
        st.subheader("Run Details")
        st.markdown(f"**Date:** {runs[selected_run_index]['timestamp']}")
        st.markdown(f"**Model ID:** {runs[selected_run_index]['model_id']}")

        # Quick actions
        st.subheader("Quick Actions")
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("📊 Export Charts"):
                st.session_state.show_export = not st.session_state.get("show_export", False)

        with export_col2:
            if st.button("📝 View Raw Data"):
                st.session_state.show_raw = not st.session_state.get("show_raw", False)

    # Main content area
    if metrics is None:
        st.error(f"Failed to load metrics from file: {selected_file}")
        return

    # Display key metrics in a clean card layout
    st.markdown("## Performance Summary")

    # Extract metrics for display
    intent_metrics = metrics.get('intent_metrics', {})
    entity_metrics = metrics.get('entity_metrics', {})
    error_analysis = metrics.get('error_analysis', {})

    intent_accuracy = intent_metrics.get('accuracy', 0)
    intent_f1 = intent_metrics.get('f1', 0)
    entity_f1 = entity_metrics.get('micro avg', {}).get('f1-score', 0)
    intent_error_rate = error_analysis.get('intent_error_rate', 0)
    entity_error_rate = error_analysis.get('entity_error_rate', 0)

    # Calculate deltas if comparing
    if compare_metrics:
        compare_intent_metrics = compare_metrics.get('intent_metrics', {})
        compare_entity_metrics = compare_metrics.get('entity_metrics', {})

        intent_accuracy_delta = intent_accuracy - compare_intent_metrics.get('accuracy', 0)
        intent_f1_delta = intent_f1 - compare_intent_metrics.get('f1', 0)
        entity_f1_delta = entity_f1 - compare_entity_metrics.get('micro avg', {}).get('f1-score', 0)
    else:
        intent_accuracy_delta = None
        intent_f1_delta = None
        entity_f1_delta = None

    # Display metrics in clean, responsive grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            custom_metric_card(
                "Intent Accuracy",
                intent_accuracy,
                delta=intent_accuracy_delta,
                help_text="Overall accuracy of intent classification",
                color="auto"
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            custom_metric_card(
                "Intent F1 Score",
                intent_f1,
                delta=intent_f1_delta,
                help_text="Weighted F1 score across all intents",
                color="auto"
            ),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            custom_metric_card(
                "Entity F1 Score",
                entity_f1,
                delta=entity_f1_delta,
                help_text="Micro-average F1 score for entity recognition",
                color="auto"
            ),
            unsafe_allow_html=True
        )

    # Create tabs for different sections
    tabs = st.tabs([
        "📈 Performance Trends",
        "🔍 Intent Analysis",
        "🏷️ Entity Analysis",
        "❌ Error Analysis"
    ])

    # Tab 1: Performance Trends
    with tabs[0]:
        st.markdown("### Performance History")

        # Load history data
        history_df = load_history()

        if history_df is not None and len(history_df) > 1:
            # Prepare data for plotting
            history_df['date_formatted'] = pd.to_datetime(history_df['date'])
            history_df = history_df.sort_values('date_formatted')

            # Create two columns
            trend_col1, trend_col2 = st.columns([3, 1])

            with trend_col1:
                # Create trend charts
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

                # Intent metrics
                ax1.plot(history_df['date_formatted'], history_df['intent_accuracy'], marker='o', label='Accuracy')
                ax1.plot(history_df['date_formatted'], history_df['intent_f1'], marker='s', label='F1')
                ax1.set_title('Intent Classification Metrics')
                ax1.set_ylabel('Score')
                ax1.set_ylim(0, 1)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()

                # Format x-axis
                ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

                # Entity metrics
                ax2.plot(history_df['date_formatted'], history_df['entity_precision'], marker='o', label='Precision')
                ax2.plot(history_df['date_formatted'], history_df['entity_recall'], marker='s', label='Recall')
                ax2.plot(history_df['date_formatted'], history_df['entity_f1'], marker='^', label='F1')
                ax2.set_title('Entity Recognition Metrics')
                ax2.set_ylabel('Score')
                ax2.set_ylim(0, 1)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()

                # Format x-axis
                ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

                plt.tight_layout()
                st.pyplot(fig)

            with trend_col2:
                # Show summary statistics and trends
                st.markdown("#### Performance Stats")

                # Calculate recent trend
                if len(history_df) >= 3:
                    recent_df = history_df.iloc[-3:]
                    intent_f1_trend = recent_df['intent_f1'].diff().mean()
                    entity_f1_trend = recent_df['entity_f1'].diff().mean()

                    # Intent F1 trend
                    trend_color = "green" if intent_f1_trend >= 0 else "red"
                    trend_icon = "↗" if intent_f1_trend >= 0 else "↘"
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Intent F1 Trend:</strong> {trend_icon} {abs(intent_f1_trend):.4f}</p>
                        <p><strong>Highest:</strong> {history_df['intent_f1'].max():.4f}</p>
                        <p><strong>Average:</strong> {history_df['intent_f1'].mean():.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Entity F1 trend
                    trend_color = "green" if entity_f1_trend >= 0 else "red"
                    trend_icon = "↗" if entity_f1_trend >= 0 else "↘"
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Entity F1 Trend:</strong> {trend_icon} {abs(entity_f1_trend):.4f}</p>
                        <p><strong>Highest:</strong> {history_df['entity_f1'].max():.4f}</p>
                        <p><strong>Average:</strong> {history_df['entity_f1'].mean():.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show data table with expandable details
                with st.expander("View History Data"):
                    st.dataframe(
                        history_df[['date', 'intent_f1', 'entity_f1', 'model_id']].sort_values('date', ascending=False),
                        hide_index=True
                    )
        else:
            st.info("Not enough history data to show trends. Run the benchmark at least twice to see performance over time.")

    # Tab 2: Intent Analysis
    with tabs[1]:
        st.markdown("### Intent Classification Analysis")

        # Get confusion matrix data
        cm = np.array(intent_metrics.get('confusion_matrix', []))
        labels = intent_metrics.get('labels', [])

        # Show confusion matrix if available
        if len(cm) > 0 and len(labels) > 0:
            # Create expander for detailed matrix if there are many classes
            if len(labels) > 15:
                with st.expander("View Detailed Confusion Matrix"):
                    st.pyplot(render_confusion_matrix(cm, labels))

                # Also show a simplified version in main view
                # Focus on most confused classes
                st.markdown("#### Top 10 Most Confused Intent Classes")

                # Find most confused classes
                np.fill_diagonal(cm, 0)  # Ignore correct predictions
                confusion_sum = np.sum(cm, axis=1)
                most_confused_indices = np.argsort(confusion_sum)[-10:]

                # Extract submatrix
                if len(most_confused_indices) > 1:
                    sub_cm = cm[np.ix_(most_confused_indices, most_confused_indices)]
                    sub_labels = [labels[i] for i in most_confused_indices]

                    st.pyplot(render_confusion_matrix(sub_cm, sub_labels))
                else:
                    st.info("No significant confusion between classes detected.")
            else:
                st.pyplot(render_confusion_matrix(cm, labels))

        # Intent class performance
        st.markdown("### Intent Class Performance")

        # Show class performance chart
        per_class_report = intent_metrics.get('per_class_report', {})
        if per_class_report:
            # Get class metrics
            fig = render_class_performance(per_class_report, "intent")
            if fig:
                st.pyplot(fig)

                # Extract low performers
                low_performers = {cls: metrics for cls, metrics in per_class_report.items()
                                if metrics['f1-score'] < 0.7 and metrics['support'] >= 3}

                # Show low performers table if any
                if low_performers:
                    st.markdown("#### Low Performing Intents")
                    st.markdown("These intents have low F1 scores and may need attention:")

                    # Convert to DataFrame for better display
                    low_df = pd.DataFrame({
                        'Intent': list(low_performers.keys()),
                        'F1 Score': [m['f1-score'] for m in low_performers.values()],
                        'Precision': [m['precision'] for m in low_performers.values()],
                        'Recall': [m['recall'] for m in low_performers.values()],
                        'Support': [m['support'] for m in low_performers.values()]
                    }).sort_values('F1 Score')

                    st.dataframe(low_df, hide_index=True)

                    # Add suggestions for improvement
                    st.markdown("""
                    <div class="card warning">
                        <h4>💡 Improvement Suggestions</h4>
                        <ul>
                            <li>Add more diverse training examples for low F1 score intents</li>
                            <li>Check for label inconsistencies in similar utterances</li>
                            <li>Consider merging very similar intents that are often confused</li>
                            <li>Review examples where precision and recall differ significantly</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No per-class performance data available.")

    # Tab 3: Entity Analysis
    with tabs[2]:
        st.markdown("### Entity Recognition Analysis")

        # Check if entity metrics are available
        if 'entity_metrics' in metrics and isinstance(entity_metrics, dict) and entity_metrics:
            # Filter out aggregate metrics
            entity_types = [key for key in entity_metrics.keys()
                          if key not in ['micro avg', 'macro avg', 'weighted avg']
                          and not (key.startswith('B-') and f'I-{key[2:]}' in entity_metrics)]

            if entity_types:
                # Extract entity type from B- prefix
                clean_entity_types = []
                for entity_type in entity_types:
                    if entity_type.startswith('B-'):
                        clean_entity_types.append(entity_type[2:])
                    elif entity_type.startswith('I-'):
                        # Skip I- tags as we already added the B- version
                        continue
                    else:
                        clean_entity_types.append(entity_type)

                # Show entity type distribution
                st.markdown("#### Entity Type Distribution")

                # Collect entity metrics
                entity_data = {
                    'Entity': [],
                    'F1 Score': [],
                    'Precision': [],
                    'Recall': [],
                    'Support': []
                }

                for entity_type in clean_entity_types:
                    # Look for B- tag first, fall back to unprefixed
                    if f'B-{entity_type}' in entity_metrics:
                        metrics_key = f'B-{entity_type}'
                    else:
                        metrics_key = entity_type

                    entity_data['Entity'].append(entity_type)
                    entity_data['F1 Score'].append(entity_metrics[metrics_key].get('f1-score', 0))
                    entity_data['Precision'].append(entity_metrics[metrics_key].get('precision', 0))
                    entity_data['Recall'].append(entity_metrics[metrics_key].get('recall', 0))
                    entity_data['Support'].append(entity_metrics[metrics_key].get('support', 0))

                # Create DataFrame for visualization
                entity_df = pd.DataFrame(entity_data)

                # Display in two columns
                entity_col1, entity_col2 = st.columns([3, 2])

                with entity_col1:
                    # Create bar chart of entity support
                    fig, ax = plt.subplots(figsize=(10, max(5, len(clean_entity_types) * 0.4)))
                    bars = ax.barh(
                        entity_df['Entity'],
                        entity_df['Support'],
                        color=plt.cm.viridis(entity_df['F1 Score'])
                    )

                    # Add F1 score as text
                    for i, (entity, f1) in enumerate(zip(entity_df['Entity'], entity_df['F1 Score'])):
                        ax.text(
                            entity_df['Support'].iloc[i] + 0.5,
                            i,
                            f'F1: {f1:.2f}',
                            va='center',
                            alpha=0.8
                        )

                    ax.set_xlabel('Number of Examples')
                    ax.set_title('Entity Types by Frequency')
                    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
                    plt.tight_layout()

                    st.pyplot(fig)

                with entity_col2:
                    # Display entity metrics table
                    st.dataframe(
                        entity_df.sort_values('F1 Score'),
                        hide_index=True
                    )

                    # Add overall metrics
                    st.markdown("#### Overall Entity Metrics")
                    if 'micro avg' in entity_metrics:
                        micro_avg = entity_metrics['micro avg']
                        st.markdown(f"""
                        <div class="card">
                            <p><strong>Micro Avg F1:</strong> {micro_avg.get('f1-score', 0):.4f}</p>
                            <p><strong>Micro Avg Precision:</strong> {micro_avg.get('precision', 0):.4f}</p>
                            <p><strong>Micro Avg Recall:</strong> {micro_avg.get('recall', 0):.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Show entity recognition error patterns if available
                entity_errors = [r for r in metrics.get('detailed_results', []) if not r.get('entities_correct', True)]

                if entity_errors:
                    st.markdown("#### Entity Recognition Error Examples")

                    # Show a sample of entity errors
                    for i, err in enumerate(entity_errors[:5]):
                        st.markdown(f"""
                        <div class="card warning">
                            <p><strong>Text:</strong> {err['text']}</p>
                            <table style="width:100%; margin-top:10px; border-collapse: collapse;">
                                <tr>
                                    <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Expected</th>
                                    <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Predicted</th>
                                </tr>
                                <tr>
                                    <td style="padding:8px; vertical-align:top; border-right:1px solid #eee;">
                                        {format_entities(err.get('true_entities', []))}
                                    </td>
                                    <td style="padding:8px; vertical-align:top;">
                                        {format_entities(err.get('pred_entities', []))}
                                    </td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                    # Show more examples in expander
                    if len(entity_errors) > 5:
                        with st.expander(f"View {len(entity_errors) - 5} More Examples"):
                            for i, err in enumerate(entity_errors[5:15]):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:10px; border-left:3px solid #ff9d4b; background:#f8f9fa;">
                                    <p><strong>Text:</strong> {err['text']}</p>
                                    <p><strong>Expected:</strong> {format_entities(err.get('true_entities', []))}</p>
                                    <p><strong>Predicted:</strong> {format_entities(err.get('pred_entities', []))}</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.info("No entity types found in the metrics.")
        else:
            st.info("No entity recognition metrics available in this benchmark run.")

    # Tab 4: Error Analysis
    with tabs[3]:
        st.markdown("### Error Analysis")

        # Get error data
        detailed_results = metrics.get('detailed_results', [])
        intent_errors = [r for r in detailed_results if not r.get('intent_correct', True)]
        entity_errors = [r for r in detailed_results if not r.get('entities_correct', True)]

        error_patterns = {}
        for err in intent_errors:
            key = (err.get('true_intent', ''), err.get('pred_intent', ''))
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(err)

        # Display error statistics
        error_stat_col1, error_stat_col2 = st.columns(2)

        with error_stat_col1:
            total_examples = len(detailed_results)
            intent_error_count = len(intent_errors)
            entity_error_count = len(entity_errors)

            # Calculate error rates
            intent_error_rate = intent_error_count / total_examples if total_examples > 0 else 0
            entity_error_rate = entity_error_count / total_examples if total_examples > 0 else 0

            # Display error rate metrics
            st.markdown(
                custom_metric_card(
                    "Intent Error Rate",
                    intent_error_rate,
                    help_text=f"{intent_error_count} errors out of {total_examples} examples",
                    color="auto"
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                custom_metric_card(
                    "Entity Error Rate",
                    entity_error_rate,
                    help_text=f"{entity_error_count} errors out of {total_examples} examples",
                    color="auto"
                ),
                unsafe_allow_html=True
            )

        with error_stat_col2:
            # Calculate confidence statistics
            if intent_errors:
                error_confidences = [err.get('confidence', 0) for err in intent_errors]
                avg_error_confidence = sum(error_confidences) / len(error_confidences)

                # Display confidence metrics
                st.markdown(
                    custom_metric_card(
                        "Avg Error Confidence",
                        avg_error_confidence,
                        help_text="Average confidence score for incorrect predictions",
                        color="auto"
                    ),
                    unsafe_allow_html=True
                )

                # Find high-confidence errors
                high_conf_errors = [err for err in intent_errors if err.get('confidence', 0) > 0.8]

                if high_conf_errors:
                    st.markdown(
                        custom_metric_card(
                            "High Confidence Errors",
                            len(high_conf_errors),
                            help_text="Errors with confidence > 0.8 (model is confidently wrong)",
                            color="orange"
                        ),
                        unsafe_allow_html=True
                    )

        # Most common error patterns
        if error_patterns:
            st.markdown("#### Most Common Error Patterns")

            # Sort patterns by frequency
            sorted_patterns = sorted(
                [(k, len(v)) for k, v in error_patterns.items()],
                key=lambda x: x[1],
                reverse=True
            )

            # Create error pattern visualization
            fig, ax = plt.subplots(figsize=(10, min(12, max(5, len(sorted_patterns[:10]) * 0.6))))

            pattern_labels = [f"{true} → {pred}" for (true, pred), _ in sorted_patterns[:10]]
            pattern_counts = [count for _, count in sorted_patterns[:10]]

            bars = ax.barh(pattern_labels, pattern_counts, color='salmon')

            # Add count as text
            for i, count in enumerate(pattern_counts):
                ax.text(count + 0.1, i, str(count), va='center')

            ax.set_xlabel('Number of Occurrences')
            ax.set_title('Most Common Error Patterns')
            ax.grid(True, linestyle='--', alpha=0.7, axis='x')

            plt.tight_layout()
            st.pyplot(fig)

            # Display error examples for top patterns
            st.markdown("#### Error Examples by Pattern")

            for i, ((true_intent, pred_intent), count) in enumerate(sorted_patterns[:5]):
                with st.expander(f"{true_intent} → {pred_intent} ({count} examples)"):
                    examples = error_patterns[(true_intent, pred_intent)]

                    for j, example in enumerate(examples[:5]):
                        st.markdown(f"""
                        <div style="margin-bottom:10px; padding:10px; border-left:3px solid #ff9d4b; background:#f8f9fa;">
                            <p><strong>Text:</strong> "{example['text']}"</p>
                            <p><strong>Confidence:</strong> {example.get('confidence', 0):.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # Recommendations based on error patterns
            st.markdown("#### Improvement Recommendations")

            # Generate tailored recommendations
            recommendations = []

            # Check for high confidence errors
            if any(err.get('confidence', 0) > 0.8 for err in intent_errors):
                recommendations.append("Review high-confidence errors - the model is confidently wrong about these examples")

            # Check for common confusion between intents
            common_confusions = [pattern for pattern, count in sorted_patterns[:3] if count > 2]
            if common_confusions:
                recommendations.append("Add more examples to distinguish between commonly confused intents")

            # Check for low support classes with errors
            low_support_intents = set()
            for err in intent_errors:
                true_intent = err.get('true_intent', '')
                if true_intent and per_class_report.get(true_intent, {}).get('support', 0) < 5:
                    low_support_intents.add(true_intent)

            if low_support_intents:
                recommendations.append(f"Add more training examples for low-support intents: {', '.join(list(low_support_intents)[:3])}")

            # Display recommendations
            if recommendations:
                for i, rec in enumerate(recommendations):
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Recommendation {i+1}:</strong> {rec}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card success">
                    <p><strong>No major issues detected.</strong> Continue monitoring performance as new data arrives.</p>
                </div>
                """, unsafe_allow_html=True)

    # Export section (conditionally shown)
    if st.session_state.get("show_export", False):
        st.markdown("---")
        st.markdown("### Export Options")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("📊 Export as HTML Report"):
                st.markdown("HTML report export functionality would be implemented here")

        with export_col2:
            if st.button("📄 Export as CSV"):
                st.markdown("CSV export functionality would be implemented here")

    # Raw data viewer (conditionally shown)
    if st.session_state.get("show_raw", False):
        st.markdown("---")
        st.markdown("### Raw Metrics Data")

        # Pretty print the JSON with syntax highlighting
        with st.expander("View Full Metrics JSON"):
            st.json(metrics)

def format_entities(entities):
"""Format entity list for display"""
if not entities:
return "<em>No entities</em>"

    result = ""
    for entity in entities:
        entity_type = entity.get('entity', 'unknown')
        entity_value = entity.get('value', '')
        result += f"<span style='background:#e6f0ff; padding:1px 5px; border-radius:3px; margin-right:5px;'>{entity_type}: <strong>{entity_value}</strong></span> "

    return result

# Initialize session state

if 'show_export' not in st.session_state:
st.session_state.show_export = False
if 'show_raw' not in st.session_state:
st.session_state.show_raw = False

# Run the app

if **name** == "**main**":
main()
Step 3.2: Create an Optimized Dashboard Launcher
Create run_dashboard.sh with efficient dependency installation and error handling:

bash
#!/bin/bash

# Configuration

DASHBOARD_SCRIPT="nlu_dashboard.py"
REQUIREMENTS_FILE="requirements-dashboard.txt"
STREAMLIT_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# Color codes for terminal output

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print styled message

print_message() {
local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if Python is installed

if ! command -v python3 &> /dev/null; then
print_message "$RED" "Error: Python 3 is not installed. Please install Python 3 first."
exit 1
fi

# Create and activate virtual environment if it doesn't exist

if [ ! -d "venv_dashboard" ]; then
print_message "$YELLOW" "Creating virtual environment for dashboard..."
    python3 -m venv venv_dashboard
    if [ $? -ne 0 ]; then
        print_message "$RED" "Failed to create virtual environment."
exit 1
fi
fi

# Create requirements file if it doesn't exist

if [ ! -f "$REQUIREMENTS_FILE" ]; then
print_message "$YELLOW" "Creating requirements file..."
    cat > "$REQUIREMENTS_FILE" << EOF
streamlit>=1.22.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
seqeval>=1.2.2
tqdm>=4.65.0
EOF
print_message "$GREEN" "Created $REQUIREMENTS_FILE"
fi

# Activate virtual environment

if [["$OSTYPE" == "msys" || "$OSTYPE" == "win32"]]; then # Windows
source venv_dashboard/Scripts/activate
else # Linux/macOS
source venv_dashboard/bin/activate
fi

if [ $? -ne 0 ]; then
print_message "$RED" "Failed to activate virtual environment."
exit 1
fi

# Check if streamlit is installed

if ! python -c "import streamlit" &> /dev/null; then
print_message "$YELLOW" "Installing required packages..."
    pip install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
print_message "$RED" "Failed to install required packages."
exit 1
fi
fi

# Check if benchmark results directory exists

if [ ! -d "benchmark_results" ]; then
print_message "$YELLOW" "Warning: No benchmark results found. Run evaluation first."
mkdir -p benchmark_results
fi

# Start the dashboard

print_message "$GREEN" "Starting NLU Benchmarking Dashboard on port $STREAMLIT_PORT..."
streamlit run "$DASHBOARD_SCRIPT" -- --server.port=$STREAMLIT_PORT --server.headless=$STREAMLIT_SERVER_HEADLESS

# Deactivate virtual environment on exit

deactivate
Step 3.3: Create a Minimal, Efficient Requirements File
Create requirements-dashboard.txt with only essential dependencies:

streamlit>=1.22.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
seqeval>=1.2.2
tqdm>=4.65.0
Step 3.4: Create a Desktop Shortcut (Optional)
For easier access, create a desktop shortcut file NLU_Dashboard.desktop (Linux/Mac) or NLU_Dashboard.bat (Windows):

For Linux/Mac (NLU_Dashboard.desktop):

[Desktop Entry]
Type=Application
Name=NLU Benchmarking Dashboard
Comment=View NLU model performance metrics
Exec=/bin/bash -c "cd %PROJECTPATH% && ./run_dashboard.sh"
Icon=utilities-terminal
Terminal=false
Categories=Development;
For Windows (NLU_Dashboard.bat):

batch
@echo off
cd /d %~dp0
start "" run_dashboard.sh
Phase 3 Validation Steps:
Installation Check: Run ./run_dashboard.sh to verify automatic installation and dependency management
Data Loading: Confirm that benchmark data loads efficiently with proper caching
Responsiveness: Test dashboard on different screen sizes to verify responsive design
Performance: Validate that large datasets don't cause performance issues
UI/UX Quality: Verify the interface follows modern design principles:
Clear information hierarchy
Intuitive navigation
Consistent visual design
Appropriate use of color and typography
Responsive layout
Feature Completeness: Ensure all required functionality is present:
Performance metrics visualization
Confusion matrix analysis
Entity recognition analysis
Error pattern identification
Historical trend tracking
Error Handling: Test with various data edge cases to verify robust error handling
Phase 4: Optimized Integration and Advanced Features
Step 4.1: Create Efficient Regression Testing System
Create a streamlined test_nlu_regression.py script with statistical significance testing and smart thresholds:

python
#!/usr/bin/env python3
"""
NLU Model Regression Test - Detects performance degradation between model versions
with configurable thresholds, statistical significance testing, and CI integration.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path

# Configure logging

logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nlu_regression')

# Type aliases for clarity

MetricsDict = Dict[str, Any]
RegressionResult = Dict[str, Any]

class RegressionTester:
"""
Smart regression testing for NLU models with statistical validation
and CI-friendly output.
"""

    # Default configuration
    DEFAULT_CONFIG = {
        'thresholds': {
            'intent_f1': 0.01,          # 1% decrease in intent F1
            'entity_f1': 0.02,          # 2% decrease in entity F1
            'accuracy': 0.01,           # 1% decrease in accuracy
            'high_impact_intents': 0.03  # 3% decrease for critical intents
        },
        'high_impact_intents': [],       # List of business-critical intents
        'significance_level': 0.05,      # p-value threshold for statistical significance
        'min_samples': 5,                # Minimum samples needed for statistical testing
        'metrics_to_track': ['intent_f1', 'entity_f1', 'accuracy'],
        'ignore_metrics': []             # Metrics to ignore in regression testing
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the regression tester with optional custom configuration.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self.DEFAULT_CONFIG.copy()

        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    if custom_config and isinstance(custom_config, dict):
                        # Update only provided values, keep defaults for others
                        self._update_nested_dict(self.config, custom_config)
                logger.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom configuration: {str(e)}")

        # Setup history tracking
        self.history_file = None
        self.history_df = None

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary recursively, preserving existing values"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def load_history(self, history_path: str) -> bool:
        """
        Load metrics history from CSV file.

        Args:
            history_path: Path to metrics history CSV

        Returns:
            bool: True if loaded successfully
        """
        self.history_file = history_path

        if not os.path.exists(history_path):
            logger.warning(f"History file not found: {history_path}")
            return False

        try:
            self.history_df = pd.read_csv(history_path)
            logger.info(f"Loaded history data with {len(self.history_df)} entries")
            return True
        except Exception as e:
            logger.error(f"Failed to load history file: {str(e)}")
            return False

    def load_metrics(self, metrics_path: str) -> Optional[MetricsDict]:
        """
        Load metrics from a JSON file.

        Args:
            metrics_path: Path to metrics JSON file

        Returns:
            MetricsDict or None if loading failed
        """
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Loaded metrics from {metrics_path}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics file {metrics_path}: {str(e)}")
            return None

    def _extract_key_metrics(self, metrics: MetricsDict) -> Dict[str, float]:
        """
        Extract key metrics from the full metrics dictionary.

        Args:
            metrics: Full metrics dictionary

        Returns:
            Dict[str, float]: Dictionary of key metrics
        """
        result = {}

        # Extract intent metrics
        intent_metrics = metrics.get('intent_metrics', {})
        result['intent_f1'] = intent_metrics.get('f1', 0.0)
        result['accuracy'] = intent_metrics.get('accuracy', 0.0)
        result['intent_precision'] = intent_metrics.get('precision', 0.0)
        result['intent_recall'] = intent_metrics.get('recall', 0.0)

        # Extract per-class intent metrics for high impact intents
        per_class = intent_metrics.get('per_class_report', {})
        for intent in self.config['high_impact_intents']:
            if intent in per_class:
                result[f'intent_{intent}_f1'] = per_class[intent].get('f1-score', 0.0)

        # Extract entity metrics
        entity_metrics = metrics.get('entity_metrics', {})
        micro_avg = entity_metrics.get('micro avg', {})
        result['entity_f1'] = micro_avg.get('f1-score', 0.0)
        result['entity_precision'] = micro_avg.get('precision', 0.0)
        result['entity_recall'] = micro_avg.get('recall', 0.0)

        # Extract error analysis
        error_analysis = metrics.get('error_analysis', {})
        result['intent_error_rate'] = error_analysis.get('intent_error_rate', 0.0)
        result['entity_error_rate'] = error_analysis.get('entity_error_rate', 0.0)

        return result

    def _get_best_previous_metrics(self) -> Dict[str, float]:
        """
        Get the best metrics from previous runs.

        Returns:
            Dict[str, float]: Dictionary of best metrics
        """
        if self.history_df is None or len(self.history_df) == 0:
            return {}

        best_metrics = {}

        # For each metric, find the best value
        for metric in self.config['metrics_to_track']:
            if metric in self.history_df.columns:
                if metric.endswith('error_rate'):
                    # For error rates, lower is better
                    best_value = self.history_df[metric].min()
                else:
                    # For all other metrics, higher is better
                    best_value = self.history_df[metric].max()
                best_metrics[metric] = best_value

        # For high impact intents
        for intent in self.config['high_impact_intents']:
            metric = f'intent_{intent}_f1'
            if metric in self.history_df.columns:
                best_metrics[metric] = self.history_df[metric].max()

        return best_metrics

    def _check_statistical_significance(
        self,
        metric_name: str,
        current_value: float,
        regression_amount: float
    ) -> bool:
        """
        Check if the regression is statistically significant.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            regression_amount: Amount of regression

        Returns:
            bool: True if the regression is statistically significant
        """
        if self.history_df is None or len(self.history_df) < self.config['min_samples']:
            # Not enough data for statistical testing
            return True

        if metric_name not in self.history_df.columns:
            return True

        # Get historical values for this metric
        historical_values = self.history_df[metric_name].dropna().values

        if len(historical_values) < self.config['min_samples']:
            return True

        # Perform one-sample t-test
        t_stat, p_value = stats.ttest_1samp(historical_values, current_value)

        # Check if regression is significant
        if p_value < self.config['significance_level'] and t_stat > 0:
            return True

        return False

    def _get_regression_detail(
        self,
        metric_name: str,
        current_value: float,
        best_value: float,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Get detailed information about a regression.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            best_value: Best historical value
            threshold: Regression threshold

        Returns:
            Dict[str, Any]: Regression details
        """
        regression_amount = best_value - current_value
        is_significant = self._check_statistical_significance(
            metric_name, current_value, regression_amount
        )

        return {
            'metric': metric_name,
            'current_value': current_value,
            'best_value': best_value,
            'regression_amount': regression_amount,
            'regression_percent': (regression_amount / best_value) * 100 if best_value > 0 else 0,
            'threshold': threshold,
            'is_significant': is_significant,
            'is_regression': regression_amount > threshold and is_significant
        }

    def check_for_regression(
        self,
        current_metrics: MetricsDict
    ) -> Tuple[bool, RegressionResult]:
        """
        Check if the current metrics show a regression compared to historical bests.

        Args:
            current_metrics: Current metrics dictionary

        Returns:
            Tuple[bool, Dict]: (has_regressed, regression_details)
        """
        # Extract key metrics from full metrics
        current_key_metrics = self._extract_key_metrics(current_metrics)

        # Get best metrics from history
        best_metrics = self._get_best_previous_metrics()

        # If no history, this is the baseline
        if not best_metrics:
            logger.info("No history found. This will be treated as the baseline.")
            return False, {
                'has_regressed': False,
                'regressions': [],
                'current_metrics': current_key_metrics,
                'best_metrics': {}
            }

        # Check each metric for regression
        regressions = []

        for metric_name, current_value in current_key_metrics.items():
            # Skip metrics in ignore list
            if metric_name in self.config['ignore_metrics']:
                continue

            # Only check metrics we're tracking
            if (metric_name not in self.config['metrics_to_track'] and
                not any(metric_name.startswith(f'intent_{intent}_f1')
                       for intent in self.config['high_impact_intents'])):
                continue

            # Get best value and threshold
            best_value = best_metrics.get(metric_name, 0.0)

            # Determine threshold based on metric type
            if any(metric_name.startswith(f'intent_{intent}_f1')
                  for intent in self.config['high_impact_intents']):
                threshold = self.config['thresholds']['high_impact_intents']
            elif metric_name in self.config['thresholds']:
                threshold = self.config['thresholds'][metric_name]
            else:
                # Default to intent_f1 threshold
                threshold = self.config['thresholds']['intent_f1']

            # Check for regression
            if current_value < best_value - threshold:
                regression_detail = self._get_regression_detail(
                    metric_name, current_value, best_value, threshold
                )

                if regression_detail['is_regression']:
                    regressions.append(regression_detail)

        # Determine if there's a regression overall
        has_regressed = len(regressions) > 0

        return has_regressed, {
            'has_regressed': has_regressed,
            'regressions': regressions,
            'current_metrics': current_key_metrics,
            'best_metrics': best_metrics
        }

    def generate_report(
        self,
        regression_result: RegressionResult,
        output_format: str = 'text'
    ) -> str:
        """
        Generate a human-readable report of regression results.

        Args:
            regression_result: Result from check_for_regression
            output_format: Format of the report ('text', 'json', 'github')

        Returns:
            str: Formatted report
        """
        if output_format == 'json':
            return json.dumps(regression_result, indent=2)

        if output_format == 'github':
            return self._generate_github_report(regression_result)

        # Default to text report
        report = []

        if regression_result['has_regressed']:
            report.append("🚨 PERFORMANCE REGRESSION DETECTED 🚨\n")

            # Add details for each regression
            for i, reg in enumerate(regression_result['regressions']):
                report.append(f"Regression {i+1}: {reg['metric']}")
                report.append(f"  Current value: {reg['current_value']:.4f}")
                report.append(f"  Previous best: {reg['best_value']:.4f}")
                report.append(f"  Decrease: {reg['regression_amount']:.4f} ({reg['regression_percent']:.2f}%)")
                report.append(f"  Threshold: {reg['threshold']:.4f}")
                report.append(f"  Statistically significant: {'Yes' if reg['is_significant'] else 'No'}")
                report.append("")
        else:
            report.append("✅ No significant performance regression detected.\n")

        # Add summary of current metrics
        report.append("Current Metrics:")
        for metric, value in sorted(regression_result['current_metrics'].items()):
            if metric in regression_result['best_metrics']:
                best = regression_result['best_metrics'][metric]
                diff = value - best
                diff_str = f" ({'↑' if diff >= 0 else '↓'}{abs(diff):.4f})"
            else:
                diff_str = " (baseline)"

            report.append(f"  {metric}: {value:.4f}{diff_str}")

        return "\n".join(report)

    def _generate_github_report(self, regression_result: RegressionResult) -> str:
        """Generate a GitHub-compatible Markdown report for GitHub Actions"""
        report = []

        if regression_result['has_regressed']:
            report.append("## 🚨 Performance Regression Detected\n")

            # Add table for regressions
            report.append("| Metric | Current | Previous Best | Decrease | Threshold |")
            report.append("| ------ | -------:| -------------:| --------:| ---------:|")

            for reg in regression_result['regressions']:
                report.append(
                    f"| {reg['metric']} | {reg['current_value']:.4f} | {reg['best_value']:.4f} | "
                    f"{reg['regression_amount']:.4f} ({reg['regression_percent']:.2f}%) | {reg['threshold']:.4f} |"
                )

            report.append("\n### Recommendations")
            report.append("- Review recent changes to training data or model architecture")
            report.append("- Check for data drift or class imbalance issues")
            report.append("- Validate that benchmark data is representative")
        else:
            report.append("## ✅ No Performance Regression\n")
            report.append("All metrics are within acceptable thresholds compared to previous best results.")

        # Add summary of current metrics
        report.append("\n### Current Metrics\n")
        report.append("| Metric | Current | Previous Best | Difference |")
        report.append("| ------ | -------:| -------------:| ----------:|")

        for metric, value in sorted(regression_result['current_metrics'].items()):
            if metric in regression_result['best_metrics']:
                best = regression_result['best_metrics'][metric]
                diff = value - best
                diff_str = f"{'↑' if diff >= 0 else '↓'}{abs(diff):.4f} ({abs(diff/best*100):.2f}%)"
            else:
                diff_str = "baseline"

            report.append(f"| {metric} | {value:.4f} | {best:.4f if metric in regression_result['best_metrics'] else 'N/A'} | {diff_str} |")

        return "\n".join(report)

def main():
"""Main entry point for regression testing"""
parser = argparse.ArgumentParser(description='Check for NLU model performance regression')
parser.add_argument('--benchmark', default='data/nlu_benchmark_data.json',
help='Path to benchmark data')
parser.add_argument('--model', default='trained_nlu_model',
help='Path to trained model')
parser.add_argument('--output-dir', default='benchmark_results',
help='Output directory for results')
parser.add_argument('--metrics-file',
help='Path to metrics file (if already generated)')
parser.add_argument('--history-file',
help='Path to metrics history file')
parser.add_argument('--config', help='Path to configuration file')
parser.add_argument('--format', choices=['text', 'json', 'github'], default='text',
help='Output format for regression report')
parser.add_argument('--ci', action='store_true',
help='Run in CI mode (exit with error code if regression detected)')
parser.add_argument('--verbose', '-v', action='store_true',
help='Enable verbose logging')

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize regression tester
    tester = RegressionTester(args.config)

    # Load metrics history
    history_file = args.history_file or os.path.join(args.output_dir, "metrics_history.csv")
    tester.load_history(history_file)

    # Get current metrics (either from file or by running evaluation)
    current_metrics = None

    if args.metrics_file and os.path.exists(args.metrics_file):
        # Load metrics from file
        current_metrics = tester.load_metrics(args.metrics_file)
    else:
        # Run evaluation to generate metrics
        logger.info("No metrics file provided. Running evaluation...")
        try:
            from evaluate_nlu import evaluate_model
            current_metrics = evaluate_model(args.benchmark, args.model, args.output_dir)
        except ImportError:
            logger.error("Could not import evaluate_nlu module. Please provide a metrics file.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            sys.exit(1)

    if not current_metrics:
        logger.error("Failed to load or generate metrics. Exiting.")
        sys.exit(1)

    # Check for regression
    has_regressed, regression_result = tester.check_for_regression(current_metrics)

    # Generate and print report
    report = tester.generate_report(regression_result, args.format)
    print(report)

    # In CI mode, exit with error code if regression detected
    if args.ci and has_regressed:
        logger.error("Performance regression detected. Failing CI.")
        sys.exit(1)

    # Success
    sys.exit(0)

if **name** == "**main**":
main()
Step 4.2: Create Intelligent Training Pipeline Integration
Create model_pipeline.py with smart model tracking and versioning:

python
#!/usr/bin/env python3
"""
Intelligent NLU Model Pipeline - Training and benchmarking pipeline
with model versioning, metadata tracking, and quality gates.
"""

import os
import sys
import json
import uuid
import shutil
import argparse
import logging
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Configure logging

logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nlu_pipeline')

class ModelPipeline:
"""
Intelligent NLU model pipeline with version tracking and quality gates.
"""

    def __init__(self, base_dir: str = "."):
        """
        Initialize the pipeline with directory structure.

        Args:
            base_dir: Base directory for the pipeline
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.benchmark_dir = self.base_dir / "benchmark_results"
        self.data_dir = self.base_dir / "data"

        # Create required directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.benchmark_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize model registry file
        self.registry_path = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load or initialize the model registry"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading model registry: {str(e)}. Creating new one.")

        # Initialize new registry
        registry = {
            "models": {},
            "current_model": None,
            "best_model": None,
            "last_updated": datetime.now().isoformat()
        }

        # Save initial registry
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return registry

    def _save_registry(self):
        """Save the model registry to disk"""
        self.registry["last_updated"] = datetime.now().isoformat()

        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _generate_model_id(self, training_data_path: str) -> str:
        """
        Generate a unique model ID based on timestamp and data hash.

        Args:
            training_data_path: Path to training data

        Returns:
            str: Unique model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a hash of the training data for deterministic versioning
        data_hash = "unknown"
        if os.path.exists(training_data_path):
            try:
                with open(training_data_path, 'rb') as f:
                    data_hash = hashlib.md5(f.read()).hexdigest()[:8]
            except Exception:
                pass

        return f"model_{timestamp}_{data_hash}"

    def _calculate_data_stats(self, data_path: str) -> Dict[str, Any]:
        """
        Calculate statistics about the training data.

        Args:
            data_path: Path to training data

        Returns:
            Dict[str, Any]: Data statistics
        """
        stats = {
            "file_size_bytes": 0,
            "example_count": 0,
            "intent_distribution": {},
            "entity_distribution": {},
            "avg_tokens_per_example": 0
        }

        if not os.path.exists(data_path):
            return stats

        try:
            with open(data_path, 'r') as f:
                data = json.load(f)

            stats["file_size_bytes"] = os.path.getsize(data_path)
            stats["example_count"] = len(data)

            total_tokens = 0

            # Calculate distributions
            for example in data:
                # Intent distribution
                intent = example.get('intent')
                if intent:
                    stats["intent_distribution"][intent] = stats["intent_distribution"].get(intent, 0) + 1

                # Entity distribution
                for entity in example.get('entities', []):
                    entity_type = entity.get('entity')
                    if entity_type:
                        stats["entity_distribution"][entity_type] = stats["entity_distribution"].get(entity_type, 0) + 1

                # Token count
                tokens = example.get('text', '').split()
                total_tokens += len(tokens)

            # Calculate average tokens
            if stats["example_count"] > 0:
                stats["avg_tokens_per_example"] = total_tokens / stats["example_count"]

        except Exception as e:
            logger.warning(f"Error calculating data stats: {str(e)}")

        return stats

    def run_command(self, command: str) -> Tuple[int, str]:
        """
        Run a shell command with output capture and logging.

        Args:
            command: Command to run

        Returns:
            Tuple[int, str]: (return_code, output)
        """
        logger.info(f"Running: {command}")

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Capture output with streaming
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            logger.debug(line)
            output_lines.append(line)

        # Wait for process to complete
        return_code = process.wait()
        output = '\n'.join(output_lines)

        return return_code, output

    def train(
        self,
        training_data_path: str,
        model_id: Optional[str] = None,
        description: str = "",
        train_args: Optional[List[str]] = None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Train a new model and track its metadata.

        Args:
            training_data_path: Path to training data
            model_id: Optional custom model ID
            description: Optional model description
            train_args: Additional arguments for train.py
            force: Force training even if data hasn't changed

        Returns:
            Tuple[bool, str]: (success, model_id)
        """
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(training_data_path)

        # Ensure model_id is safe for filesystem
        model_id = model_id.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Define model directory
        model_dir = self.models_dir / model_id

        # Check if model with this ID already exists
        if os.path.exists(model_dir) and not force:
            logger.warning(f"Model with ID {model_id} already exists. Use force=True to overwrite.")
            return False, model_id

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Calculate data statistics
        data_stats = self._calculate_data_stats(training_data_path)

        # Create metadata file
        metadata = {
            "model_id": model_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "training_data": {
                "path": str(training_data_path),
                "stats": data_stats
            },
            "train_args": train_args or [],
            "metrics": None,
            "status": "training"
        }

        # Save initial metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Build training command
        train_cmd = [
            "python", "train.py",
            "--data", str(training_data_path),
            "--output", str(model_dir)
        ]

        # Add additional arguments
        if train_args:
            train_cmd.extend(train_args)

        # Run training
        train_command = " ".join(train_cmd)
        return_code, output = self.run_command(train_command)

        # Update metadata with training results
        metadata["train_output"] = output

        if return_code != 0:
            metadata["status"] = "failed"
            logger.error(f"Training failed with code {return_code}")

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return False, model_id

        # Training succeeded
        metadata["status"] = "trained"

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry["models"][model_id] = {
            "path": str(model_dir),
            "created_at": metadata["created_at"],
            "description": description,
            "status": "trained"
        }

        # Set as current model
        self.registry["current_model"] = model_id
        self._save_registry()

        logger.info(f"Successfully trained model: {model_id}")
        return True, model_id

    def benchmark(
        self,
        model_id: Optional[str] = None,
        benchmark_data_path: Optional[str] = None,
        benchmark_args: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Benchmark a model and store its metrics.

        Args:
            model_id: Model ID to benchmark (uses current if None)
            benchmark_data_path: Path to benchmark data
            benchmark_args: Additional arguments for evaluate_nlu.py

        Returns:
            Tuple[bool, Dict]: (success, metrics)
        """
        # Use current model if not specified
        if model_id is None:
            model_id = self.registry.get("current_model")
            if model_id is None:
                logger.error("No current model found. Train a model first.")
                return False, {}

        # Check if model exists
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False, {}

        # Get model directory
        model_dir = Path(self.registry["models"][model_id]["path"])

        # Default benchmark data path
        if benchmark_data_path is None:
            benchmark_data_path = str(self.data_dir / "nlu_benchmark_data.json")

        # Create benchmark output directory
        benchmark_output_dir = self.benchmark_dir / model_id
        os.makedirs(benchmark_output_dir, exist_ok=True)

        # Build benchmark command
        benchmark_cmd = [
            "python", "evaluate_nlu.py",
            "--benchmark", str(benchmark_data_path),
            "--model", str(model_dir),
            "--output", str(benchmark_output_dir)
        ]

        # Add additional arguments
        if benchmark_args:
            benchmark_cmd.extend(benchmark_args)

        # Run benchmark
        benchmark_command = " ".join(benchmark_cmd)
        return_code, output = self.run_command(benchmark_command)

        if return_code != 0:
            logger.error(f"Benchmarking failed with code {return_code}")
            return False, {}

        # Find the metrics file
        metrics_files = list(benchmark_output_dir.glob("metrics_*.json"))
        if not metrics_files:
            logger.error("No metrics file found after benchmarking.")
            return False, {}

        # Get the latest metrics file
        latest_metrics_file = max(metrics_files, key=os.path.getctime)

        # Load metrics
        try:
            with open(latest_metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics file: {str(e)}")
            return False, {}

        # Update model metadata with metrics
        metadata_path = model_dir / "metadata.json"

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata["metrics"] = metrics
            metadata["benchmarked_at"] = datetime.now().isoformat()
            metadata["benchmark_data_path"] = str(benchmark_data_path)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update model metadata with metrics: {str(e)}")

        # Update registry
        self.registry["models"][model_id]["benchmarked_at"] = datetime.now().isoformat()
        self.registry["models"][model_id]["metrics_path"] = str(latest_metrics_file)

        # Extract key metrics for registry
        key_metrics = {}
        intent_metrics = metrics.get("intent_metrics", {})
        entity_metrics = metrics.get("entity_metrics", {})

        key_metrics["intent_f1"] = intent_metrics.get("f1", 0.0)
        key_metrics["accuracy"] = intent_metrics.get("accuracy", 0.0)

        if isinstance(entity_metrics, dict) and "micro avg" in entity_metrics:
            key_metrics["entity_f1"] = entity_metrics["micro avg"].get("f1-score", 0.0)

        self.registry["models"][model_id]["key_metrics"] = key_metrics

        # Check if this is the best model
        if self._is_best_model(model_id):
            self.registry["best_model"] = model_id
            logger.info(f"New best model: {model_id}")

        self._save_registry()

        logger.info(f"Successfully benchmarked model: {model_id}")
        return True, metrics

    def _is_best_model(self, model_id: str) -> bool:
        """
        Check if the given model is the best model based on metrics.

        Args:
            model_id: Model ID to check

        Returns:
            bool: True if this is the best model
        """
        # Get current best model
        best_model_id = self.registry.get("best_model")

        # If no best model yet, this is the best
        if best_model_id is None:
            return True

        # Get metrics for both models
        current_metrics = self.registry["models"][model_id].get("key_metrics", {})
        best_metrics = self.registry["models"].get(best_model_id, {}).get("key_metrics", {})

        # Compare intent_f1 (primary metric)
        current_f1 = current_metrics.get("intent_f1", 0.0)
        best_f1 = best_metrics.get("intent_f1", 0.0)

        return current_f1 > best_f1

    def check_regression(
        self,
        model_id: Optional[str] = None,
        config_path: Optional[str] = None,
        ci_mode: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a model has regressed compared to the best model.

        Args:
            model_id: Model ID to check (uses current if None)
            config_path: Path to regression test configuration
            ci_mode: Whether to run in CI mode

        Returns:
            Tuple[bool, Dict]: (has_regressed, details)
        """
        # Use current model if not specified
        if model_id is None:
            model_id = self.registry.get("current_model")
            if model_id is None:
                logger.error("No current model found. Train a model first.")
                return False, {}

        # Check if model exists
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False, {}

        # Get metrics path
        metrics_path = self.registry["models"][model_id].get("metrics_path")
        if not metrics_path:
            logger.error(f"No metrics found for model {model_id}. Run benchmark first.")
            return False, {}

        # Build regression test command
        regression_cmd = [
            "python", "test_nlu_regression.py",
            "--metrics-file", str(metrics_path),
            "--history-file", str(self.benchmark_dir / "metrics_history.csv")
        ]

        # Add config path if provided
        if config_path:
            regression_cmd.extend(["--config", str(config_path)])

        # Add CI mode if enabled
        if ci_mode:
            regression_cmd.append("--ci")

        # Add format
        regression_cmd.extend(["--format", "json"])

        # Run regression test
        regression_command = " ".join(regression_cmd)
        return_code, output = self.run_command(regression_command)

        # Parse JSON output
        try:
            # Find JSON in output (may have other text mixed in)
            json_start = output.find('{')
            json_end = output.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                regression_result = json.loads(json_str)
            else:
                # Fallback for when output is not proper JSON
                regression_result = {"has_regressed": return_code != 0}
        except Exception as e:
            logger.warning(f"Failed to parse regression test output: {str(e)}")
            regression_result = {"has_regressed": return_code != 0}

        # Update model metadata with regression result
        try:
            model_dir = Path(self.registry["models"][model_id]["path"])
            metadata_path = model_dir / "metadata.json"

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata["regression_test"] = {
                "result": regression_result,
                "ran_at": datetime.now().isoformat()
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update model metadata with regression result: {str(e)}")

        has_regressed = regression_result.get("has_regressed", False)

        if has_regressed:
            logger.warning(f"Model {model_id} has regressed!")
        else:
            logger.info(f"Model {model_id} has not regressed.")

        return has_regressed, regression_result

    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all models in the registry.

        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        model_list = []

        for model_id, model_data in self.registry["models"].items():
            # Determine if this is the current or best model
            is_current = model_id == self.registry.get("current_model")
            is_best = model_id == self.registry.get("best_model")

            # Add model to list
            model_list.append({
                "model_id": model_id,
                "path": model_data["path"],
                "created_at": model_data.get("created_at"),
                "description": model_data.get("description", ""),
                "status": model_data.get("status", "unknown"),
                "key_metrics": model_data.get("key_metrics", {}),
                "is_current": is_current,
                "is_best": is_best
            })

        # Sort by creation date (newest first)
        model_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return model_list

    def set_current_model(self, model_id: str) -> bool:
        """
        Set the current model for inference.

        Args:
            model_id: Model ID to set as current

        Returns:
            bool: Success
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False

        # Update registry
        self.registry["current_model"] = model_id
        self._save_registry()

        logger.info(f"Set current model to: {model_id}")
        return True

    def export_model(self, model_id: Optional[str] = None, export_dir: str = "export") -> bool:
        """
        Export a model for deployment.

        Args:
            model_id: Model ID to export (uses current if None)
            export_dir: Directory to export to

        Returns:
            bool: Success
        """
        # Use current model if not specified
        if model_id is None:
            model_id = self.registry.get("current_model")
            if model_id is None:
                logger.error("No current model found. Train a model first.")
                return False

        # Check if model exists
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False

        # Get model directory
        model_dir = Path(self.registry["models"][model_id]["path"])

        # Create export directory
        export_path = Path(export_dir)
        os.makedirs(export_path, exist_ok=True)

        try:
            # Copy model files
            shutil.copytree(
                model_dir,
                export_path / model_id,
                dirs_exist_ok=True
            )

            # Create symlink to 'latest'
            latest_link = export_path / "latest"
            if os.path.exists(latest_link):
                os.remove(latest_link)

            # Create relative symlink
            os.symlink(model_id, latest_link, target_is_directory=True)

            logger.info(f"Exported model {model_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            return False

    def run_full_pipeline(
        self,
        training_data_path: str,
        benchmark_data_path: str,
        description: str = "",
        train_args: Optional[List[str]] = None,
        benchmark_args: Optional[List[str]] = None,
        regression_config_path: Optional[str] = None,
        export_dir: Optional[str] = None,
        fail_on_regression: bool = False
    ) -> Tuple[bool, str]:
        """
        Run the full model pipeline: train, benchmark, regression test, export.

        Args:
            training_data_path: Path to training data
            benchmark_data_path: Path to benchmark data
            description: Model description
            train_args: Additional arguments for train.py
            benchmark_args: Additional arguments for evaluate_nlu.py
            regression_config_path: Path to regression test configuration
            export_dir: Directory to export to if successful
            fail_on_regression: Whether to fail if regression detected

        Returns:
            Tuple[bool, str]: (success, model_id)
        """
        logger.info("Starting full model pipeline")

        # Step 1: Train model
        logger.info("Step 1: Training model")
        train_success, model_id = self.train(
            training_data_path=training_data_path,
            description=description,
            train_args=train_args
        )

        if not train_success:
            logger.error("Pipeline failed at training step")
            return False, model_id

        # Step 2: Benchmark model
        logger.info("Step 2: Benchmarking model")
        benchmark_success, metrics = self.benchmark(
            model_id=model_id,
            benchmark_data_path=benchmark_data_path,
            benchmark_args=benchmark_args
        )

        if not benchmark_success:
            logger.error("Pipeline failed at benchmarking step")
            return False, model_id

        # Step 3: Regression test
        logger.info("Step 3: Running regression test")
        has_regressed, regression_details = self.check_regression(
            model_id=model_id,
            config_path=regression_config_path
        )

        if has_regressed and fail_on_regression:
            logger.error("Pipeline failed due to regression")
            return False, model_id

        # Step 4: Export model if requested
        if export_dir:
            logger.info("Step 4: Exporting model")
            export_success = self.export_model(
                model_id=model_id,
                export_dir=export_dir
            )

            if not export_success:
                logger.error("Pipeline failed at export step")
                return False, model_id

        logger.info(f"Full pipeline completed successfully. Model ID: {model_id}")
        return True, model_id

def main():
"""Main entry point for the model pipeline"""
parser = argparse.ArgumentParser(description='NLU Model Pipeline')
parser.add_argument('--base-dir', default=".", help='Base directory for pipeline')

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--model-id', help='Custom model ID')
    train_parser.add_argument('--description', default="", help='Model description')
    train_parser.add_argument('--force', action='store_true', help='Force training')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark a model')
    benchmark_parser.add_argument('--model-id', help='Model ID to benchmark (uses current if not specified)')
    benchmark_parser.add_argument('--data', help='Path to benchmark data')

    # Regression command
    regression_parser = subparsers.add_parser('regression', help='Run regression test')
    regression_parser.add_argument('--model-id', help='Model ID to test (uses current if not specified)')
    regression_parser.add_argument('--config', help='Path to regression test configuration')
    regression_parser.add_argument('--ci', action='store_true', help='Run in CI mode')

    # List command
    list_parser = subparsers.add_parser('list', help='List all models')
    list_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export a model')
    export_parser.add_argument('--model-id', help='Model ID to export (uses current if not specified)')
    export_parser.add_argument('--export-dir', default='export', help='Directory to export to')

    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--training-data', required=True, help='Path to training data')
    pipeline_parser.add_argument('--benchmark-data', required=True, help='Path to benchmark data')
    pipeline_parser.add_argument('--description', default="", help='Model description')
    pipeline_parser.add_argument('--regression-config', help='Path to regression test configuration')
    pipeline_parser.add_argument('--export-dir', help='Directory to export to if successful')
    pipeline_parser.add_argument('--fail-on-regression', action='store_true', help='Fail if regression detected')

    # Parse arguments
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ModelPipeline(args.base_dir)

    # Run command
    if args.command == 'train':
        success, model_id = pipeline.train(
            training_data_path=args.data,
            model_id=args.model_id,
            description=args.description,
            force=args.force
        )

        if success:
            print(f"Successfully trained model: {model_id}")
        else:
            print(f"Failed to train model: {model_id}")
            sys.exit(1)

    elif args.command == 'benchmark':
        success, metrics = pipeline.benchmark(
            model_id=args.model_id,
            benchmark_data_path=args.data
        )

        if success:
            # Print key metrics
            intent_metrics = metrics.get("intent_metrics", {})
            entity_metrics = metrics.get("entity_metrics", {})

            print("\nKey Metrics:")
            print(f"Intent Accuracy: {intent_metrics.get('accuracy', 0):.4f}")
            print(f"Intent F1 Score: {intent_metrics.get('f1', 0):.4f}")

            if isinstance(entity_metrics, dict) and "micro avg" in entity_metrics:
                print(f"Entity F1 Score: {entity_metrics['micro avg'].get('f1-score', 0):.4f}")
        else:
            print("Failed to benchmark model")
            sys.exit(1)

    elif args.command == 'regression':
        has_regressed, details = pipeline.check_regression(
            model_id=args.model_id,
            config_path=args.config,
            ci_mode=args.ci
        )

        if args.ci and has_regressed:
            sys.exit(1)

    elif args.command == 'list':
        model_list = pipeline.get_model_list()

        if args.json:
            print(json.dumps(model_list, indent=2))
        else:
            print("\nModel Registry:")
            print(f"{'Model ID':<30} {'Status':<10} {'Intent F1':<10} {'Created At':<20} {'Description'}")
            print("-" * 80)

            for model in model_list:
                # Mark current and best models
                marker = ""
                if model["is_current"] and model["is_best"]:
                    marker = "[CURRENT,BEST] "
                elif model["is_current"]:
                    marker = "[CURRENT] "
                elif model["is_best"]:
                    marker = "[BEST] "

                model_id = model["model_id"]
                status = model["status"]
                intent_f1 = model.get("key_metrics", {}).get("intent_f1", 0)
                created_at = model.get("created_at", "")[:19]  # Truncate to date+time
                description = model.get("description", "")

                print(f"{marker}{model_id:<20} {status:<10} {intent_f1:<10.4f} {created_at:<20} {description}")

    elif args.command == 'export':
        success = pipeline.export_model(
            model_id=args.model_id,
            export_dir=args.export_dir
        )

        if not success:
            print("Failed to export model")
            sys.exit(1)

    elif args.command == 'pipeline':
        success, model_id = pipeline.run_full_pipeline(
            training_data_path=args.training_data,
            benchmark_data_path=args.benchmark_data,
            description=args.description,
            regression_config_path=args.regression_config,
            export_dir=args.export_dir,
            fail_on_regression=args.fail_on_regression
        )

        if success:
            print(f"Pipeline completed successfully. Model ID: {model_id}")
        else:
            print(f"Pipeline failed. Model ID: {model_id}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

if **name** == "**main**":
main()
Step 4.3: Create CI/CD Integration Configuration
Create .github/workflows/nlu-ci.yml for GitHub Actions integration:

yaml
name: NLU Model Pipeline

on:
push:
branches: [ main ]
paths: - 'data/**' - '_.py' - 'requirements_.txt'
pull_request:
branches: [ main ]
paths: - 'data/**' - '_.py' - 'requirements_.txt'
workflow_dispatch:
inputs:
train_new_model:
description: 'Train a new model'
required: false
default: 'false'
type: boolean

jobs:
test:
name: Test NLU Model Pipeline
runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        if [ -f requirements-dashboard.txt ]; then pip install -r requirements-dashboard.txt; fi

    - name: Run tests
      run: |
        # Run basic sanity tests
        python -c "from inference import NLUInferencer; print('NLUInferencer can be imported')"
        python -c "from evaluate_nlu import evaluate_model; print('evaluate_nlu can be imported')"

    - name: Set up model directory
      run: |
        mkdir -p models
        mkdir -p benchmark_results

    # Only run if manually triggered with train_new_model=true or data changes
    - name: Train and benchmark model
      if: ${{ github.event_name == 'workflow_dispatch' && github.event.inputs.train_new_model == 'true' || contains(github.event.head_commit.message, '[train]') }}
      run: |
        python model_pipeline.py pipeline \
          --training-data data/nlu_training_data.json \
          --benchmark-data data/nlu_benchmark_data.json \
          --description "CI build ${{ github.run_id }}" \
          --fail-on-regression

    # Always run regression test on the current model
    - name: Test current model for regression
      if: ${{ always() }}
      run: |
        python model_pipeline.py regression --ci
      continue-on-error: true

    # Create model report as artifact
    - name: Generate model report
      if: ${{ always() }}
      run: |
        python model_pipeline.py list --json > model_report.json

    - name: Upload model report
      if: ${{ always() }}
      uses: actions/upload-artifact@v3
      with:
        name: model-report
        path: model_report.json

Step 4.4: Create Concise Documentation
Create docs/nlu_benchmarking.md with essential documentation:

markdown

# NLU Model Benchmarking System

This document provides a concise overview of the NLU benchmarking system, its components, and usage.

## Components

The NLU benchmarking system consists of:

1. **Core Evaluation** (`evaluate_nlu.py`): Measures model performance against a benchmark dataset
2. **Metrics Tracking** (`metrics_history.csv`): Historical performance tracking
3. **Visualization Dashboard** (`nlu_dashboard.py`): Interactive metrics visualization
4. **Regression Testing** (`test_nlu_regression.py`): Detects performance degradation
5. **Model Pipeline** (`model_pipeline.py`): Manages the entire model lifecycle

## Quick Start

### Running a Benchmark

````bash
# Basic benchmark
python evaluate_nlu.py --benchmark data/nlu_benchmark_data.json

# View results in dashboard
./run_dashboard.sh
Using the Model Pipeline
bash
# Train a new model
python model_pipeline.py train --data data/nlu_training_data.json --description "My new model"

# Benchmark the model
python model_pipeline.py benchmark

# Check for regression
python model_pipeline.py regression

# Run full pipeline
python model_pipeline.py pipeline \
  --training-data data/nlu_training_data.json \
  --benchmark-data data/nlu_benchmark_data.json \
  --description "Complete pipeline" \
  --export-dir export
Using the Dashboard
The dashboard provides a comprehensive view of model performance with:

Performance history tracking
Intent and entity analysis
Error pattern detection
Model comparison
Interpreting Results
Key Metrics
Intent F1 Score: Overall quality of intent classification (primary metric)
Entity F1 Score: Overall quality of entity recognition
Accuracy: Proportion of correctly classified intents
Error Rates: Proportion of examples with errors
Understanding Regressions
A regression occurs when a new model performs worse than previous models. The system detects regressions by:

Comparing against historical best metrics
Applying statistical significance testing
Using configurable thresholds for different metrics
Best Practices
Maintain a stable benchmark dataset that represents real-world queries
Track performance over time to identify trends and issues
Prioritize high-impact intents that are critical to your application
Set appropriate thresholds for regression detection
Review error patterns to identify common issues
Directory Structure
├── data/
│   ├── nlu_training_data.json    # Training data
│   └── nlu_benchmark_data.json   # Benchmark data
├── benchmark_results/            # Benchmark results and history
├── models/                       # Trained models
├── model_registry.json           # Model metadata
├── evaluate_nlu.py               # Evaluation script
├── model_pipeline.py             # Model lifecycle management
├── test_nlu_regression.py        # Regression testing
└── nlu_dashboard.py              # Visualization dashboard
Advanced Configuration
The regression testing system supports custom configuration via a YAML file:

yaml
thresholds:
  intent_f1: 0.01          # 1% decrease in intent F1
  entity_f1: 0.02          # 2% decrease in entity F1
  high_impact_intents: 0.03 # 3% decrease for critical intents

high_impact_intents:
  - towing_request_tow_urgent
  - roadside_emergency_situation

significance_level: 0.05   # p-value threshold (95% confidence)
Pass the configuration file with --config:

bash
python test_nlu_regression.py --config regression_config.yaml

## Phase 4 Validation Steps:
1. **Regression Testing**: Test the regression system with both improved and degraded models
   ```bash
   python test_nlu_regression.py --metrics-file benchmark_results/metrics_latest.json
Model Pipeline: Test the model pipeline with a small dataset
bash
python model_pipeline.py pipeline \
  --training-data data/nlu_training_data.json \
  --benchmark-data data/nlu_benchmark_data.json \
  --description "Test pipeline"
CI Integration: If using GitHub, push a small change to verify the workflow triggers
Documentation: Confirm the documentation is clear, concise and includes all key workflows
Verify Integration: Ensure all components work together seamlessly:
Train a model with the pipeline
Benchmark the model
View results in the dashboard
Check for regressions
Export the model
Final Validation Checklist
Before considering the implementation complete, verify that:

☑️ The benchmark dataset exists and is well-structured
☑️ Entity evaluation with BIO tag conversion works correctly
☑️ All metrics are calculated and saved properly
☑️ The Streamlit dashboard shows all relevant information
☑️ Performance history is tracked and visualized
☑️ Error analysis provides actionable insights
☑️ Regression testing detects performance degradation
☑️ Integration with training workflow is seamless
☑️ Documentation is clear and comprehensive
Next Steps
After completing this implementation, consider:

Expanding the benchmark dataset with new examples
Adding more advanced visualizations to the dashboard
Implementing automated error categorization
Setting up CI/CD integration for continuous benchmarking
Creating model comparison tools to evaluate different architectures
This benchmarking system will provide valuable insights into your NLU model's performance, help detect regressions early, and guide your improvement efforts.

````
