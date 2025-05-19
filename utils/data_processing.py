"""
Data processing utilities for the NLU Benchmarking Dashboard
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
from collections import Counter, defaultdict

def extract_intent_distributions(metrics):
    """
    Extract intent distribution data from metrics
    
    Args:
        metrics: Dictionary containing intent metrics
        
    Returns:
        Dictionary with intent distributions
    """
    result = {}
    
    if 'intent_metrics' not in metrics or 'per_class_report' not in metrics['intent_metrics']:
        return result
        
    per_class = metrics['intent_metrics']['per_class_report']
    
    # Get total support
    # Only count support for actual intents, not the averages
    total_support = 0
    for intent, data in per_class.items():
        if intent not in ['micro avg', 'macro avg', 'weighted avg']:
            support = data.get('support', 0)
            total_support += support
    
    # Extract data
    intent_data = []
    for intent, data in per_class.items():
        if intent not in ['micro avg', 'macro avg', 'weighted avg']:
            support = data.get('support', 0)
            percentage = (support / total_support) * 100 if total_support > 0 else 0
            
            intent_data.append({
                'intent': intent,
                'f1_score': data.get('f1-score', 0),
                'precision': data.get('precision', 0),
                'recall': data.get('recall', 0),
                'support': support,
                'percentage': percentage
            })
    
    # Sort by F1 score
    intent_data.sort(key=lambda x: x['f1_score'])
    
    result['intent_data'] = intent_data
    result['total_support'] = total_support
    
    return result

def process_confusion_matrix(confusion_matrix, labels):
    """
    Process confusion matrix data for visualization
    
    Args:
        confusion_matrix: 2D array of confusion matrix
        labels: List of class labels
        
    Returns:
        Processed confusion matrix data
    """
    if len(confusion_matrix) == 0 or len(labels) == 0:
        return {}
        
    cm = np.array(confusion_matrix)
    
    # Calculate per-class metrics from confusion matrix
    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    
    # Find most confused pairs
    confused_pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true': labels[i],
                    'predicted': labels[j],
                    'count': int(cm[i, j]),
                    'true_total': int(np.sum(cm[i, :])),
                    'percentage': float(cm[i, j] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0)
                })
                
    # Sort by count, descending
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        'confusion_matrix': cm.tolist(),
        'normalized_matrix': (cm.astype('float') / cm.sum(axis=1, keepdims=True)).tolist(),
        'labels': labels,
        'true_positives': true_positives.tolist(),
        'false_positives': false_positives.tolist(),
        'false_negatives': false_negatives.tolist(),
        'confused_pairs': confused_pairs[:10]  # Top 10 most confused pairs
    }

def analyze_errors(detailed_results):
    """
    Analyze error patterns in detailed results
    
    Args:
        detailed_results: List of detailed prediction results
        
    Returns:
        Dictionary with error analysis data
    """
    if not detailed_results:
        return {}
        
    # Extract errors
    errors = [r for r in detailed_results if not r.get('intent_correct', True)]
    correct = [r for r in detailed_results if r.get('intent_correct', True)]
    
    # Error confidence stats
    error_confidences = [e.get('confidence', 0) for e in errors if 'confidence' in e]
    correct_confidences = [c.get('confidence', 0) for c in correct if 'confidence' in c]
    
    # Group errors by patterns
    error_patterns = defaultdict(list)
    for error in errors:
        key = (error.get('true_intent', 'Unknown'), error.get('pred_intent', 'Unknown'))
        error_patterns[key].append(error)
        
    # Sort patterns by frequency
    sorted_patterns = sorted(
        [(k, len(v)) for k, v in error_patterns.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Format patterns
    formatted_patterns = []
    for (true, pred), count in sorted_patterns:
        formatted_patterns.append({
            'true_intent': true,
            'predicted_intent': pred,
            'count': count,
            'percentage': (count / len(errors)) * 100 if errors else 0
        })
    
    # Calculate high confidence errors
    high_confidence_threshold = 0.8
    high_confidence_errors = [e for e in errors if e.get('confidence', 0) >= high_confidence_threshold]
    
    return {
        'total_examples': len(detailed_results),
        'error_count': len(errors),
        'error_rate': len(errors) / len(detailed_results) if detailed_results else 0,
        'avg_error_confidence': np.mean(error_confidences) if error_confidences else 0,
        'avg_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
        'high_confidence_errors': len(high_confidence_errors),
        'high_confidence_error_rate': len(high_confidence_errors) / len(errors) if errors else 0,
        'error_patterns': formatted_patterns,
        'errors': errors,
        'correct': correct
    }

def process_history_data(history_df):
    """
    Process historical data for trending and analysis
    
    Args:
        history_df: DataFrame containing historical metrics
        
    Returns:
        Dictionary with processed historical data
    """
    if history_df.empty:
        return {}
        
    # Ensure timestamp is in datetime format
    if 'timestamp' in history_df.columns:
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    elif 'date' in history_df.columns:
        history_df['timestamp'] = pd.to_datetime(history_df['date'])
        
    # Sort by timestamp
    history_df = history_df.sort_values('timestamp')
    
    # Calculate changes between consecutive runs
    changes = {}
    
    # Columns to track changes for
    metrics_columns = ['intent_accuracy', 'intent_f1', 'entity_f1']
    
    for col in metrics_columns:
        if col in history_df.columns and len(history_df) > 1:
            diffs = history_df[col].diff()
            significant_changes = []
            
            for i, (idx, row) in enumerate(history_df.iterrows()):
                if i > 0 and abs(diffs[idx]) > 0.05:  # Significant change threshold
                    prev_timestamp = history_df.iloc[i-1]['timestamp']
                    significant_changes.append({
                        'timestamp': row['timestamp'],
                        'prev_timestamp': prev_timestamp,
                        'value': row[col],
                        'prev_value': history_df.iloc[i-1][col],
                        'change': diffs[idx],
                        'model_id': row.get('model_id', 'Unknown'),
                        'prev_model_id': history_df.iloc[i-1].get('model_id', 'Unknown')
                    })
            
            changes[col] = significant_changes
    
    # Calculate trends
    trends = {}
    for col in metrics_columns:
        if col in history_df.columns and len(history_df) >= 3:
            # Use last 3 points to determine trend
            last_values = history_df[col].iloc[-3:].tolist()
            
            if last_values[-1] > last_values[-2] > last_values[-3]:
                trend = 'increasing'
            elif last_values[-1] < last_values[-2] < last_values[-3]:
                trend = 'decreasing'
            else:
                trend = 'stable'
                
            trends[col] = {
                'trend': trend,
                'current': last_values[-1],
                'previous': last_values[-2],
                'change': last_values[-1] - last_values[-2]
            }
    
    return {
        'history_df': history_df,
        'changes': changes,
        'trends': trends,
        'latest_metrics': history_df.iloc[-1].to_dict() if not history_df.empty else {}
    }

def extract_entity_metrics(metrics):
    """
    Extract and process entity metrics
    
    Args:
        metrics: Dictionary containing entity metrics
        
    Returns:
        Dictionary with processed entity metrics
    """
    result = {}
    
    if 'entity_metrics' not in metrics:
        return result
        
    entity_metrics = metrics['entity_metrics']
    
    # Extract per-entity metrics
    if 'report' in entity_metrics:
        entity_data = []
        for entity, data in entity_metrics['report'].items():
            if entity not in ['micro avg', 'macro avg', 'weighted avg']:
                entity_data.append({
                    'entity': entity,
                    'f1_score': data.get('f1-score', 0),
                    'precision': data.get('precision', 0),
                    'recall': data.get('recall', 0),
                    'support': data.get('support', 0)
                })
                
        # Sort by F1 score
        entity_data.sort(key=lambda x: x['f1_score'])
        result['entity_data'] = entity_data
        
    # Add aggregate metrics
    for avg_type in ['micro avg', 'macro avg', 'weighted avg']:
        if 'report' in entity_metrics and avg_type in entity_metrics['report']:
            result[avg_type] = entity_metrics['report'][avg_type]
    
    return result

def load_available_models():
    """
    Load available model metadata from benchmark runs.
    
    Returns:
        List of dictionaries with model metadata.
    """
    # Check if benchmark_results directory exists
    benchmark_dir = "benchmark_results"
    if not os.path.exists(benchmark_dir):
        return []
    
    # Find all metrics files
    metric_files = sorted(glob.glob(os.path.join(benchmark_dir, "metrics_*.json")), reverse=True)
    models = []

    for file in metric_files:
        # Extract timestamp from filename
        timestamp = os.path.basename(file).replace("metrics_", "").replace(".json", "")
        try:
            # Format timestamp for display
            formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

            # Try to extract model_id
            model_id = "Unknown"
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    model_id = data.get("model_id", "Unknown")
            except:
                pass

            models.append({
                "id": model_id,
                "file": file,
                "date": formatted_time,
                "timestamp": timestamp,
                "version": model_id.split("_")[-1] if "_" in model_id else "Unknown"
            })
        except Exception as e:
            # Skip files with invalid naming
            continue

    return models


def load_model_metrics(model_id):
    """
    Load metrics for a specific model.
    
    Args:
        model_id: ID of the model to load metrics for
        
    Returns:
        Dictionary containing model metrics or None if not found
    """
    # Get available models
    models = load_available_models()
    
    # Find the model with the given ID
    model_file = None
    for model in models:
        if model["id"] == model_id:
            model_file = model["file"]
            break
    
    if not model_file:
        return None
    
    # Load metrics
    try:
        with open(model_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {str(e)}")
        return None 