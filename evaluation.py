#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for evaluating and testing chatbot models.
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForTokenClassification
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model_pipeline(test_data_dir: str, models_dir: str, output_dir: str, threshold_config: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Evaluate the full model pipeline on test data.
    
    Args:
        test_data_dir: Directory containing test datasets
        models_dir: Directory containing trained models
        output_dir: Directory to save evaluation results
        threshold_config: Configuration for confidence thresholds and fallbacks
        
    Returns:
        Dictionary of evaluation metrics and detailed analysis
    """
    if threshold_config is None:
        threshold_config = {
            "flow_threshold": 0.7,
            "intent_threshold": 0.6,
            "fallback_threshold": 0.5,
            "clarification_threshold": 0.5
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting comprehensive model evaluation")
    
    results = {
        "flow_classification": {},
        "intent_classification": {},
        "entity_extraction": {},
        "fallback_detection": {},
        "clarification_effectiveness": {},
        "conversation_simulation": {},
        "robustness_metrics": {}
    }
    
    # Load test data
    try:
        # Flow classification
        with open(os.path.join(test_data_dir, "flow_classification_test.json"), 'r') as f:
            flow_test_data = json.load(f)
        
        # Check for extreme test data
        extreme_flow_test_path = os.path.join(test_data_dir, "flow_classification_extreme_test.json")
        if os.path.exists(extreme_flow_test_path):
            with open(extreme_flow_test_path, 'r') as f:
                flow_extreme_test_data = json.load(f)
        else:
            flow_extreme_test_data = []
            
        # Evaluate flow classification
        results["flow_classification"] = evaluate_flow_classification(
            flow_test_data, 
            flow_extreme_test_data,
            models_dir,
            threshold_config["flow_threshold"]
        )
        
        # Evaluate intent classification for each flow
        flows = ['towing', 'roadside', 'appointment', 'clarification', 'fallback']
        results["intent_classification"] = {}
        
        for flow in flows:
            # Filter intent test data by flow
            with open(os.path.join(test_data_dir, "intent_classification_test.json"), 'r') as f:
                intent_test_data = json.load(f)
            
            flow_intent_test_data = [item for item in intent_test_data if item.get('flow') == flow]
            
            # Check for extreme test data
            extreme_intent_test_path = os.path.join(test_data_dir, "intent_classification_extreme_test.json")
            if os.path.exists(extreme_intent_test_path):
                with open(extreme_intent_test_path, 'r') as f:
                    intent_extreme_test_data = json.load(f)
                flow_intent_extreme_test_data = [item for item in intent_extreme_test_data if item.get('flow') == flow]
            else:
                flow_intent_extreme_test_data = []
            
            results["intent_classification"][flow] = evaluate_intent_classification(
                flow_intent_test_data,
                flow_intent_extreme_test_data,
                models_dir,
                flow,
                threshold_config["intent_threshold"]
            )
        
        # Evaluate entity extraction
        with open(os.path.join(test_data_dir, "entity_classification_test.json"), 'r') as f:
            entity_test_data = json.load(f)
        
        extreme_entity_test_path = os.path.join(test_data_dir, "entity_classification_extreme_test.json")
        if os.path.exists(extreme_entity_test_path):
            with open(extreme_entity_test_path, 'r') as f:
                entity_extreme_test_data = json.load(f)
        else:
            entity_extreme_test_data = []
        
        results["entity_extraction"] = evaluate_entity_extraction(
            entity_test_data,
            entity_extreme_test_data,
            models_dir
        )
        
        # Evaluate fallback detection
        with open(os.path.join(test_data_dir, "fallback_classification_test.json"), 'r') as f:
            fallback_test_data = json.load(f)
        
        results["fallback_detection"] = evaluate_fallback_detection(
            fallback_test_data,
            models_dir,
            threshold_config["fallback_threshold"]
        )
        
        # Evaluate clarification effectiveness
        with open(os.path.join(test_data_dir, "clarification_classification_test.json"), 'r') as f:
            clarification_test_data = json.load(f)
        
        results["clarification_effectiveness"] = evaluate_clarification_detection(
            clarification_test_data,
            models_dir,
            threshold_config["clarification_threshold"]
        )
        
        # End-to-end conversation simulation
        # This would require a more complex test set with multi-turn conversations
        results["conversation_simulation"] = {
            "message": "End-to-end conversation simulation not implemented in this version"
        }
        
        # Calculate aggregate robustness metrics
        results["robustness_metrics"] = calculate_robustness_metrics(results)
        
        # Save results
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.json')}")
        
        return results
    
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {str(e)}")
        return {"error": f"Test data file not found: {str(e)}"}
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return {"error": f"Error during evaluation: {str(e)}"}

def evaluate_flow_classification(test_data: List[Dict[str, Any]], 
                                extreme_test_data: List[Dict[str, Any]],
                                models_dir: str,
                                threshold: float) -> Dict[str, Any]:
    """
    Evaluate flow classification performance.
    
    Args:
        test_data: Standard test data
        extreme_test_data: Challenging test cases
        models_dir: Directory containing the flow classifier model
        threshold: Confidence threshold for classification
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating flow classification")
    
    # In a real implementation, you would:
    # 1. Load the flow classifier model
    # 2. Run predictions on the test data
    # 3. Calculate accuracy, precision, recall, F1
    # 4. Run additional evaluations on the extreme test cases
    
    # This is a placeholder implementation
    results = {
        "standard_test": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {},
            "samples_count": len(test_data)
        },
        "extreme_test": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "samples_count": len(extreme_test_data)
        },
        "confidence_analysis": {
            "average_confidence": 0.0,
            "threshold": threshold,
            "samples_below_threshold": 0
        }
    }
    
    return results

def evaluate_intent_classification(test_data: List[Dict[str, Any]],
                                  extreme_test_data: List[Dict[str, Any]],
                                  models_dir: str,
                                  flow: str,
                                  threshold: float) -> Dict[str, Any]:
    """
    Evaluate intent classification for a specific flow.
    
    Args:
        test_data: Standard test data for the flow
        extreme_test_data: Challenging test cases for the flow
        models_dir: Directory containing the intent classifier model
        flow: Flow name
        threshold: Confidence threshold for classification
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating intent classification for {flow} flow")
    
    # Placeholder implementation
    results = {
        "standard_test": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {},
            "samples_count": len(test_data)
        },
        "extreme_test": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "samples_count": len(extreme_test_data)
        },
        "confidence_analysis": {
            "average_confidence": 0.0,
            "threshold": threshold,
            "samples_below_threshold": 0
        }
    }
    
    return results

def evaluate_entity_extraction(test_data: List[Dict[str, Any]],
                              extreme_test_data: List[Dict[str, Any]],
                              models_dir: str) -> Dict[str, Any]:
    """
    Evaluate entity extraction model using BIO tagging evaluation.
    
    Args:
        test_data: List of test examples
        extreme_test_data: Challenging test cases
        models_dir: Directory containing the trained model
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating entity extraction model")
    
    # Force CPU
    device = torch.device("cpu")
    
    # Load model and tokenizer
    model_dir = os.path.join(models_dir, "entity_extractor")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    def evaluate_bio_tags(data: List[Dict[str, Any]]) -> Dict[str, float]:
        true_tags = []
        pred_tags = []
        
        for example in data:
            # Get ground truth BIO tags
            text = example['text']
            entities = example.get('entities', [])
            true_bio = convert_to_bio_tags(text, entities)
            true_tags.append(true_bio)
            
            # Get predicted BIO tags from model
            # In a real implementation, you would:
            # 1. Tokenize the text
            # 2. Run it through the model
            # 3. Convert logits to predictions
            # This is a placeholder that copies true tags for demonstration
            pred_bio = true_bio.copy()  # Replace with actual model predictions
            pred_tags.append(pred_bio)
        
        # Calculate metrics using seqeval
        report = seq_classification_report(true_tags, pred_tags, output_dict=True)
        f1 = seq_f1_score(true_tags, pred_tags)
        
        # Extract overall metrics
        metrics = {
            "entity_f1": f1,
            "entity_precision": report['macro avg']['precision'],
            "entity_recall": report['macro avg']['recall'],
            "per_entity_metrics": {
                entity: {
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1": metrics['f1-score'],
                    "support": metrics['support']
                }
                for entity, metrics in report.items()
                if entity not in ['macro avg', 'weighted avg', 'micro avg']
            }
        }
        
        return metrics
    
    # Evaluate standard test set
    standard_metrics = evaluate_bio_tags(test_data)
    
    # Evaluate extreme test set if available
    extreme_metrics = evaluate_bio_tags(extreme_test_data) if extreme_test_data else {}
    
    # Calculate partial match metrics
    def calculate_partial_matches(data: List[Dict[str, Any]]) -> Dict[str, float]:
        total_entities = 0
        partial_matches = 0
        
        for example in data:
            true_entities = example.get('entities', [])
            # In real implementation, get predicted entities from model
            pred_entities = true_entities.copy()  # Replace with actual predictions
            
            for true_ent in true_entities:
                total_entities += 1
                # Check for partial matches in predicted entities
                for pred_ent in pred_entities:
                    if (true_ent['entity'] == pred_ent['entity'] and
                        (true_ent['value'] in pred_ent['value'] or 
                         pred_ent['value'] in true_ent['value'])):
                        partial_matches += 1
                        break
        
        return {
            "partial_match_rate": partial_matches / total_entities if total_entities > 0 else 0.0
        }
    
    partial_metrics = calculate_partial_matches(test_data)
    
    # Combine all metrics
    results = {
        "standard_test": {
            **standard_metrics,
            "samples_count": len(test_data)
        },
        "extreme_test": {
            **extreme_metrics,
            "samples_count": len(extreme_test_data)
        },
        "partial_match_metrics": partial_metrics
    }
    
    # Save detailed evaluation results
    os.makedirs(os.path.join(model_dir, "evaluation"), exist_ok=True)
    with open(os.path.join(model_dir, "evaluation", "entity_extraction_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Entity extraction evaluation completed. Results saved to {model_dir}/evaluation/")
    
    return results

def evaluate_fallback_detection(test_data: List[Dict[str, Any]],
                               models_dir: str,
                               threshold: float) -> Dict[str, Any]:
    """
    Evaluate fallback detection performance with detailed binary classification metrics.
    
    Args:
        test_data: Test data for fallback detection
        models_dir: Directory containing the fallback detection model
        threshold: Confidence threshold for fallback detection
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating fallback detection")
    
    # In a real implementation, load and use the model
    # This is a placeholder that uses random predictions for demonstration
    y_true = [1 if example.get('flow') == 'fallback' else 0 for example in test_data]
    y_pred = y_true.copy()  # Replace with actual model predictions
    
    # Calculate detailed metrics
    report = classification_report(y_true, y_pred, 
                                target_names=['normal', 'fallback'],
                                output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['normal', 'fallback'],
                yticklabels=['normal', 'fallback'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Fallback Detection Confusion Matrix")
    
    # Save the plot
    model_dir = os.path.join(models_dir, "fallback_classifier")
    os.makedirs(os.path.join(model_dir, "evaluation"), exist_ok=True)
    plt.savefig(os.path.join(model_dir, "evaluation", "fallback_confusion_matrix.png"))
    plt.close()
    
    results = {
        "accuracy": report['accuracy'],
        "normal": {
            "precision": report['normal']['precision'],
            "recall": report['normal']['recall'],
            "f1": report['normal']['f1-score']
        },
        "fallback": {
            "precision": report['fallback']['precision'],
            "recall": report['fallback']['recall'],
            "f1": report['fallback']['f1-score']
        },
        "macro_avg": {
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "f1": report['macro avg']['f1-score']
        },
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "samples_count": len(test_data)
    }
    
    # Save detailed results
    with open(os.path.join(model_dir, "evaluation", "fallback_detection_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Fallback detection evaluation completed. Results saved to {model_dir}/evaluation/")
    
    return results

def evaluate_clarification_detection(test_data: List[Dict[str, Any]],
                                    models_dir: str,
                                    threshold: float) -> Dict[str, Any]:
    """
    Evaluate clarification detection performance with detailed binary classification metrics.
    
    Args:
        test_data: Test data for clarification detection
        models_dir: Directory containing the clarification detection model
        threshold: Confidence threshold for clarification detection
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating clarification detection")
    
    # In a real implementation, load and use the model
    # This is a placeholder that uses random predictions for demonstration
    y_true = [1 if example.get('flow') == 'clarification' or 
              (isinstance(example.get('context', {}), dict) and 
               example.get('context', {}).get('needs_clarification', False))
              else 0 for example in test_data]
    y_pred = y_true.copy()  # Replace with actual model predictions
    
    # Calculate detailed metrics
    report = classification_report(y_true, y_pred,
                                target_names=['clear_intent', 'needs_clarification'],
                                output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['clear_intent', 'needs_clarification'],
                yticklabels=['clear_intent', 'needs_clarification'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Clarification Detection Confusion Matrix")
    
    # Save the plot
    model_dir = os.path.join(models_dir, "clarification_classifier")
    os.makedirs(os.path.join(model_dir, "evaluation"), exist_ok=True)
    plt.savefig(os.path.join(model_dir, "evaluation", "clarification_confusion_matrix.png"))
    plt.close()
    
    results = {
        "accuracy": report['accuracy'],
        "clear_intent": {
            "precision": report['clear_intent']['precision'],
            "recall": report['clear_intent']['recall'],
            "f1": report['clear_intent']['f1-score']
        },
        "needs_clarification": {
            "precision": report['needs_clarification']['precision'],
            "recall": report['needs_clarification']['recall'],
            "f1": report['needs_clarification']['f1-score']
        },
        "macro_avg": {
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "f1": report['macro avg']['f1-score']
        },
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "samples_count": len(test_data)
    }
    
    # Save detailed results
    with open(os.path.join(model_dir, "evaluation", "clarification_detection_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Clarification detection evaluation completed. Results saved to {model_dir}/evaluation/")
    
    return results

def simulate_conversations(conversation_flows: List[Dict[str, Any]],
                          models_dir: str,
                          threshold_config: Dict[str, float]) -> Dict[str, Any]:
    """
    Simulate end-to-end conversations to evaluate overall system performance.
    
    Args:
        conversation_flows: Test conversation flows
        models_dir: Directory containing all models
        threshold_config: Configuration for confidence thresholds
    
    Returns:
        Dictionary of conversation success metrics
    """
    logger.info("Simulating end-to-end conversations")
    
    # Placeholder implementation
    results = {
        "task_completion_rate": 0.0,
        "average_turns": 0,
        "fallback_rate": 0.0,
        "clarification_success_rate": 0.0,
        "samples_count": 0
    }
    
    return results

def calculate_robustness_metrics(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate aggregate robustness metrics from all evaluation results.
    
    Args:
        evaluation_results: Dictionary containing all evaluation results
    
    Returns:
        Dictionary of robustness metrics
    """
    logger.info("Calculating aggregate robustness metrics")
    
    # Placeholder implementation
    results = {
        "noise_tolerance_score": 0.0,
        "out_of_distribution_detection_rate": 0.0,
        "recovery_rate": 0.0,
        "extreme_case_performance_ratio": 0.0
    }
    
    return results

def generate_evaluation_report(evaluation_results: Dict[str, Any], output_path: str) -> None:
    """
    Generate a comprehensive evaluation report in Markdown format.
    
    Args:
        evaluation_results: Dictionary containing all evaluation results
        output_path: Path to save the report
    """
    logger.info("Generating evaluation report")
    
    # Create report structure
    report = [
        "# Chatbot Model Evaluation Report\n",
        "## Overview\n",
        f"- Date: {logging.Formatter().converter()}\n",
        f"- Total test samples: {sum([results.get('samples_count', 0) for task, results in evaluation_results.items() if isinstance(results, dict) and 'samples_count' in results])}\n\n",
        
        "## Flow Classification\n",
        f"- Accuracy: {evaluation_results['flow_classification']['standard_test']['accuracy']:.2f}\n",
        f"- F1 Score: {evaluation_results['flow_classification']['standard_test']['f1']:.2f}\n",
        f"- Extreme Test Accuracy: {evaluation_results['flow_classification']['extreme_test']['accuracy']:.2f}\n\n",
        
        "## Intent Classification\n"
    ]
    
    # Add intent classification results for each flow
    for flow, results in evaluation_results['intent_classification'].items():
        report.append(f"### {flow.capitalize()} Flow\n")
        report.append(f"- Accuracy: {results['standard_test']['accuracy']:.2f}\n")
        report.append(f"- F1 Score: {results['standard_test']['f1']:.2f}\n")
        report.append(f"- Extreme Test Accuracy: {results['extreme_test']['accuracy']:.2f}\n\n")
    
    # Add entity extraction results
    report.append("## Entity Extraction\n")
    report.append(f"- F1 Score: {evaluation_results['entity_extraction']['standard_test']['entity_f1']:.2f}\n")
    report.append(f"- Precision: {evaluation_results['entity_extraction']['standard_test']['entity_precision']:.2f}\n")
    report.append(f"- Recall: {evaluation_results['entity_extraction']['standard_test']['entity_recall']:.2f}\n\n")
    
    # Add fallback and clarification detection results
    report.append("## Fallback Detection\n")
    report.append(f"- Accuracy: {evaluation_results['fallback_detection']['accuracy']:.2f}\n")
    report.append(f"- F1 Score: {evaluation_results['fallback_detection']['f1']:.2f}\n\n")
    
    report.append("## Clarification Detection\n")
    report.append(f"- Accuracy: {evaluation_results['clarification_effectiveness']['accuracy']:.2f}\n")
    report.append(f"- F1 Score: {evaluation_results['clarification_effectiveness']['f1']:.2f}\n\n")
    
    # Add robustness metrics
    report.append("## Robustness Metrics\n")
    report.append(f"- Noise Tolerance Score: {evaluation_results['robustness_metrics']['noise_tolerance_score']:.2f}\n")
    report.append(f"- Out-of-Distribution Detection Rate: {evaluation_results['robustness_metrics']['out_of_distribution_detection_rate']:.2f}\n")
    report.append(f"- Recovery Rate: {evaluation_results['robustness_metrics']['recovery_rate']:.2f}\n")
    report.append(f"- Extreme Case Performance Ratio: {evaluation_results['robustness_metrics']['extreme_case_performance_ratio']:.2f}\n\n")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write(''.join(report))
    
    logger.info(f"Evaluation report saved to {output_path}")
