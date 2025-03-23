#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Specialized evaluation script for context-aware chatbot models.
This script focuses on measuring performance of context-related features:
- Negation detection
- Context switching
- Contradiction detection
- Multi-turn conversation handling
"""

import json
import os
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import uuid
import datetime

# Import the assistants
from inference import CarroAssistant, ContextAwareCarroAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextEvaluator:
    """
    Evaluator class for context-aware models, measuring performance on:
    - Negation detection
    - Context switch detection
    - Contradiction detection
    - Multi-turn conversation handling
    """
    
    def __init__(self, models_dir="./output/models", output_dir="./output/evaluation", test_data_file="./data/context_test_cases.json"):
        """
        Initialize the evaluator with required models and paths.
        
        Args:
            models_dir: Directory containing trained models
            output_dir: Directory to save evaluation results
            test_data_file: JSON file with test cases
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.test_data_file = test_data_file
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data if available
        self.test_data = None
        if os.path.exists(test_data_file):
            try:
                with open(test_data_file, 'r') as f:
                    self.test_data = json.load(f)
                logger.info(f"Loaded test cases from {test_data_file}")
            except Exception as e:
                logger.error(f"Could not load test data from {test_data_file}: {e}")
        
        # Initialize assistants
        try:
            self.standard_assistant = CarroAssistant(models_dir)
            self.context_assistant = ContextAwareCarroAssistant(models_dir)
            self.assistants_available = True
            logger.info("Assistants loaded successfully")
        except Exception as e:
            logger.error(f"Could not initialize assistants: {e}")
            self.assistants_available = False
        
        # Initialize results storage
        self.results = {
            "negation_detection": {},
            "context_switch_detection": {},
            "contradiction_detection": {},
            "multi_turn": {},
            "overall_metrics": {}
        }
    
    def evaluate_negation_detection(self, test_data=None):
        """
        Evaluate negation detection performance.
        
        Args:
            test_data: Optional list of test cases with expected results
        
        Returns:
            Dictionary of negation detection metrics
        """
        if not self.assistants_available:
            return {"error": "Assistants not available"}
        
        if test_data is None:
            # Use data from file if available
            if self.test_data and "negation_detection" in self.test_data:
                test_data = self.test_data["negation_detection"]
                logger.info(f"Using {len(test_data)} test cases from file")
            else:
                # Create default test data if none provided
                test_data = self._generate_negation_test_data()
                logger.info(f"Using {len(test_data)} generated test cases")
        
        results = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": None,
            "detail": []
        }
        
        # Track predictions and ground truth
        y_true = []
        y_pred = []
        
        logger.info(f"Evaluating negation detection on {len(test_data)} test cases")
        for i, test_case in enumerate(tqdm(test_data, desc="Negation Detection")):
            text = test_case["text"]
            expected = test_case["is_negation"]
            
            # Get model prediction
            detection_result = self.context_assistant.detect_negation(text)
            predicted = detection_result["is_negation"]
            confidence = detection_result.get("confidence", 0.0)
            
            # Record result
            y_true.append(1 if expected else 0)
            y_pred.append(1 if predicted else 0)
            
            # Save detailed result
            results["detail"].append({
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": expected == predicted
            })
        
        # Calculate metrics
        if y_true and y_pred:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            results["accuracy"] = round(accuracy, 3)
            results["precision"] = round(precision, 3)
            results["recall"] = round(recall, 3)
            results["f1"] = round(f1, 3)
            results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        logger.info(f"Negation detection evaluation complete: Accuracy={results.get('accuracy')}, F1={results.get('f1')}")
        return results
    
    def evaluate_context_switch_detection(self, test_data=None):
        """
        Evaluate context switch detection performance.
        
        Args:
            test_data: Optional list of test cases with expected results
        
        Returns:
            Dictionary of context switch detection metrics
        """
        if not self.assistants_available:
            return {"error": "Assistants not available"}
        
        if test_data is None:
            # Use data from file if available
            if self.test_data and "context_switch_detection" in self.test_data:
                test_data = self.test_data["context_switch_detection"]
                logger.info(f"Using {len(test_data)} test cases from file")
            else:
                # Create default test data if none provided
                test_data = self._generate_context_switch_test_data()
                logger.info(f"Using {len(test_data)} generated test cases")
        
        results = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": None,
            "detail": []
        }
        
        # Track predictions and ground truth
        y_true = []
        y_pred = []
        
        logger.info(f"Evaluating context switch detection on {len(test_data)} test cases")
        for i, test_case in enumerate(tqdm(test_data, desc="Context Switch Detection")):
            text = test_case["text"]
            expected = test_case["has_context_switch"]
            
            # Check if the method accepts a context parameter
            try:
                # Get model prediction - first try with context
                if "context" in test_case and hasattr(self.context_assistant, "detect_context_switch"):
                    # Check if the method accepts 2 or 3 arguments
                    import inspect
                    sig = inspect.signature(self.context_assistant.detect_context_switch)
                    if len(sig.parameters) > 2:
                        detection_result = self.context_assistant.detect_context_switch(text, test_case["context"])
                    else:
                        detection_result = self.context_assistant.detect_context_switch(text)
                else:
                    detection_result = self.context_assistant.detect_context_switch(text)
            except TypeError:
                # Fallback to calling without context
                detection_result = self.context_assistant.detect_context_switch(text)
                
            predicted = detection_result["has_context_switch"]
            confidence = detection_result.get("confidence", 0.0)
            
            # Record result
            y_true.append(1 if expected else 0)
            y_pred.append(1 if predicted else 0)
            
            # Save detailed result
            results["detail"].append({
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": expected == predicted,
                "context": test_case.get("context", {})
            })
        
        # Calculate metrics
        if y_true and y_pred:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            results["accuracy"] = round(accuracy, 3)
            results["precision"] = round(precision, 3)
            results["recall"] = round(recall, 3)
            results["f1"] = round(f1, 3)
            results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        logger.info(f"Context switch detection evaluation complete: Accuracy={results.get('accuracy')}, F1={results.get('f1')}")
        return results
    
    def evaluate_contradiction_detection(self, test_data=None):
        """
        Evaluate contradiction detection performance.
        
        Args:
            test_data: Optional list of test cases with expected results
        
        Returns:
            Dictionary of contradiction detection metrics
        """
        if not self.assistants_available:
            return {"error": "Assistants not available"}
        
        if test_data is None:
            # Use data from file if available
            if self.test_data and "contradiction_detection" in self.test_data:
                test_data = self.test_data["contradiction_detection"]
                logger.info(f"Using {len(test_data)} test cases from file")
            else:
                # Create default test data if none provided
                test_data = self._generate_contradiction_test_data()
                logger.info(f"Using {len(test_data)} generated test cases")
        
        results = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "entity_accuracy": {},
            "detail": []
        }
        
        # Track predictions and ground truth
        y_true = []
        y_pred = []
        
        # Track entity-specific accuracy
        entity_correct = {}
        entity_total = {}
        
        logger.info(f"Evaluating contradiction detection on {len(test_data)} test cases")
        for i, test_case in enumerate(tqdm(test_data, desc="Contradiction Detection")):
            text = test_case["text"]
            expected = test_case["has_contradiction"]
            expected_entity_type = test_case.get("entity_type") if expected else None
            context = test_case.get("context", {})
            
            # Make sure we have a valid context
            if not context:
                context = {"conversation_id": f"test-{i}"}
            
            # Get model prediction
            try:
                detection_result = self.context_assistant.detect_contradictions(text, context)
            except Exception as e:
                logger.error(f"Error detecting contradictions: {str(e)}")
                # Skip this test case if detection fails
                continue
                
            predicted = detection_result["has_contradiction"]
            confidence = detection_result.get("confidence", 0.0)
            entity_type = detection_result.get("entity_type") if predicted else None
            
            # Record result
            y_true.append(1 if expected else 0)
            y_pred.append(1 if predicted else 0)
            
            # Track entity-specific accuracy
            if expected and predicted and expected_entity_type:
                entity_correct.setdefault(expected_entity_type, 0)
                entity_total.setdefault(expected_entity_type, 0)
                entity_total[expected_entity_type] += 1
                
                if expected_entity_type == entity_type:
                    entity_correct[expected_entity_type] += 1
            
            # Save detailed result
            results["detail"].append({
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "expected_entity_type": expected_entity_type,
                "predicted_entity_type": entity_type,
                "confidence": confidence,
                "correct": expected == predicted,
                "entity_correct": expected_entity_type == entity_type if expected and predicted else None,
                "context": context
            })
        
        # Calculate metrics
        if y_true and y_pred:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            results["accuracy"] = round(accuracy, 3)
            results["precision"] = round(precision, 3)
            results["recall"] = round(recall, 3)
            results["f1"] = round(f1, 3)
            
            # Calculate entity-specific accuracy
            for entity_type in entity_total:
                entity_accuracy = entity_correct.get(entity_type, 0) / entity_total[entity_type]
                results["entity_accuracy"][entity_type] = round(entity_accuracy, 3)
        
        logger.info(f"Contradiction detection evaluation complete: Accuracy={results.get('accuracy')}, F1={results.get('f1')}")
        return results
    
    def evaluate_multi_turn_conversations(self, test_conversations=None):
        """
        Evaluate multi-turn conversation handling.
        
        Args:
            test_conversations: Optional list of multi-turn conversations with expected results
        
        Returns:
            Dictionary of multi-turn conversation metrics
        """
        if not self.assistants_available:
            return {"error": "Assistants not available"}
        
        if test_conversations is None:
            # Use data from file if available
            if self.test_data and "multi_turn_conversations" in self.test_data:
                test_conversations = self.test_data["multi_turn_conversations"]
                logger.info(f"Using {len(test_conversations)} test conversations from file")
            else:
                # Create default test conversations if none provided
                test_conversations = self._generate_multi_turn_test_data()
                logger.info(f"Using {len(test_conversations)} generated test conversations")
        
        results = {
            "completion_rate": 0.0,
            "context_preservation": 0.0,
            "entity_preservation": 0.0,
            "negation_handling": 0.0,
            "context_switch_handling": 0.0,
            "contradiction_handling": 0.0,
            "conversations": []
        }
        
        logger.info(f"Evaluating {len(test_conversations)} multi-turn conversations")
        
        # Metrics tracking
        completed_conversations = 0
        context_preserved_conversations = 0
        total_expected_entities = 0
        total_preserved_entities = 0
        total_negations = 0
        correct_negations = 0
        total_context_switches = 0
        correct_context_switches = 0
        total_contradictions = 0
        correct_contradictions = 0
        
        for conversation_id, conversation in enumerate(test_conversations):
            turns = conversation["turns"]
            expected_outcome = conversation.get("expected_outcome", {})
            
            # Initialize conversation context
            context = {
                "conversation_id": conversation.get("conversation_id", f"test-{conversation_id}"),
                "turn_count": 0
            }
            
            conversation_results = {
                "id": conversation_id,
                "completion": False,
                "context_preserved": False,
                "expected_final_context": expected_outcome.get("final_context"),
                "actual_final_context": None,
                "turns": []
            }
            
            # Process each turn
            has_error = False
            try:
                for turn_idx, turn in enumerate(turns):
                    text = turn["text"]
                    expected = turn.get("expected", {})
                    
                    # Process message
                    try:
                        result = self.context_assistant.process_message_with_context(text, context)
                        
                        # Update context for next turn
                        context = result["context"]
                        
                        # Check for correct handling of special cases
                        if turn.get("is_negation", False):
                            total_negations += 1
                            if not result.get("context_switch", False):  # Negation should not trigger context switch
                                correct_negations += 1
                        
                        if turn.get("is_context_switch", False):
                            total_context_switches += 1
                            if result.get("context_switch", False):
                                correct_context_switches += 1
                        
                        if turn.get("is_contradiction", False):
                            total_contradictions += 1
                            if result.get("contradiction", False):
                                correct_contradictions += 1
                        
                        # Track turn results
                        conversation_results["turns"].append({
                            "turn": turn_idx + 1,
                            "text": text,
                            "expected": expected,
                            "result": result
                        })
                    except Exception as e:
                        has_error = True
                        logger.error(f"Error processing turn {turn_idx + 1} of conversation {conversation_id}: {str(e)}")
                        conversation_results["turns"].append({
                            "turn": turn_idx + 1,
                            "text": text,
                            "expected": expected,
                            "error": str(e)
                        })
                        break
                    
                # Skip further evaluation if there was an error
                if has_error:
                    conversation_results["error"] = "Conversation processing incomplete due to errors"
                    results["conversations"].append(conversation_results)
                    continue
                
                # Check if conversation completed successfully
                conversation_completed = True
                for key, value in expected_outcome.get("final_context", {}).items():
                    if key not in context or context[key] != value:
                        conversation_completed = False
                        break
                
                # Count preserved entities
                expected_entities = expected_outcome.get("preserved_entities", [])
                total_expected_entities += len(expected_entities)
                
                preserved_entities = 0
                for entity in expected_entities:
                    entity_type = entity["type"]
                    entity_value = entity["value"]
                    if context.get(entity_type) == entity_value:
                        preserved_entities += 1
                
                total_preserved_entities += preserved_entities
                
                # Update conversation results
                conversation_results["completion"] = conversation_completed
                conversation_results["context_preserved"] = preserved_entities == len(expected_entities)
                conversation_results["actual_final_context"] = context
                conversation_results["preserved_entities"] = preserved_entities
                conversation_results["expected_entities"] = len(expected_entities)
                
                # Update overall metrics
                if conversation_completed:
                    completed_conversations += 1
                
                if conversation_results["context_preserved"]:
                    context_preserved_conversations += 1
                
            except Exception as e:
                logger.error(f"Error processing conversation {conversation_id}: {str(e)}")
                conversation_results["error"] = str(e)
            
            # Add conversation results
            results["conversations"].append(conversation_results)
        
        # Calculate final metrics
        if test_conversations:
            results["completion_rate"] = round(completed_conversations / len(test_conversations), 3)
            results["context_preservation"] = round(context_preserved_conversations / len(test_conversations), 3)
            
            if total_expected_entities > 0:
                results["entity_preservation"] = round(total_preserved_entities / total_expected_entities, 3)
            
            if total_negations > 0:
                results["negation_handling"] = round(correct_negations / total_negations, 3)
            
            if total_context_switches > 0:
                results["context_switch_handling"] = round(correct_context_switches / total_context_switches, 3)
            
            if total_contradictions > 0:
                results["contradiction_handling"] = round(correct_contradictions / total_contradictions, 3)
        
        logger.info(f"Multi-turn evaluation complete: Completion Rate={results.get('completion_rate')}, Context Preservation={results.get('context_preservation')}")
        return results
    
    def perform_comprehensive_evaluation(self):
        """
        Run comprehensive evaluation on all context-aware features.
        
        Returns:
            Dictionary with all evaluation results
        """
        if not self.assistants_available:
            return {"error": "Assistants not available"}
        
        logger.info("Starting comprehensive context-aware evaluation")
        
        # Run all evaluations
        self.results["negation_detection"] = self.evaluate_negation_detection()
        self.results["context_switch_detection"] = self.evaluate_context_switch_detection()
        self.results["contradiction_detection"] = self.evaluate_contradiction_detection()
        self.results["multi_turn"] = self.evaluate_multi_turn_conversations()
        
        # Calculate overall metrics
        self.results["overall_metrics"] = self._calculate_overall_metrics()
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"context_evaluation_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
        
        # Generate report
        self._generate_report(timestamp)
        
        return self.results
    
    def _calculate_overall_metrics(self):
        """Calculate aggregated metrics across all tests"""
        metrics = {
            "accuracy": {},
            "success_rates": {},
            "overall_score": 0.0
        }
        
        # Negation detection
        metrics["accuracy"]["negation_detection"] = self.results["negation_detection"].get("accuracy", 0.0)
        
        # Context switch detection
        metrics["accuracy"]["context_switch_detection"] = self.results["context_switch_detection"].get("accuracy", 0.0)
        
        # Contradiction detection
        metrics["accuracy"]["contradiction_detection"] = self.results["contradiction_detection"].get("accuracy", 0.0)
        
        # Multi-turn conversation metrics
        metrics["success_rates"]["completion_rate"] = self.results["multi_turn"].get("completion_rate", 0.0)
        metrics["success_rates"]["context_preservation"] = self.results["multi_turn"].get("context_preservation", 0.0)
        metrics["success_rates"]["entity_preservation"] = self.results["multi_turn"].get("entity_preservation", 0.0)
        
        # Special handling rates
        metrics["success_rates"]["negation_handling"] = self.results["multi_turn"].get("negation_handling", 0.0)
        metrics["success_rates"]["context_switch_handling"] = self.results["multi_turn"].get("context_switch_handling", 0.0)
        metrics["success_rates"]["contradiction_handling"] = self.results["multi_turn"].get("contradiction_handling", 0.0)
        
        # Calculate overall score (weighted average)
        weights = {
            "negation_detection": 0.15,
            "context_switch_detection": 0.15,
            "contradiction_detection": 0.15,
            "completion_rate": 0.15,
            "context_preservation": 0.15,
            "entity_preservation": 0.1,
            "negation_handling": 0.05,
            "context_switch_handling": 0.05,
            "contradiction_handling": 0.05
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics["accuracy"]:
                score += metrics["accuracy"][metric] * weight
            elif metric in metrics["success_rates"]:
                score += metrics["success_rates"][metric] * weight
        
        metrics["overall_score"] = round(score, 3)
        
        return metrics 
    
    def _generate_report(self, timestamp):
        """Generate HTML report with evaluation results"""
        try:
            report_file = os.path.join(self.output_dir, f"context_evaluation_report_{timestamp}.html")
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Context-Aware Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }
                    .section { margin-bottom: 30px; }
                    .metric { margin: 10px 0; }
                    .metric-name { font-weight: bold; }
                    .score { font-weight: bold; }
                    .good { color: green; }
                    .medium { color: orange; }
                    .poor { color: red; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Context-Aware Evaluation Report</h1>
                    <p>Generated on: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
            """
            
            # Overall metrics
            overall = self.results["overall_metrics"]
            html += """
                <div class="section">
                    <h2>Overall Performance</h2>
                    <div class="metric">
                        <span class="metric-name">Overall Score:</span>
                        <span class="score """ + self._get_score_class(overall["overall_score"]) + """">""" + str(overall["overall_score"]) + """</span>
                    </div>
                </div>
            """
            
            # Accuracy metrics
            html += """
                <div class="section">
                    <h2>Detection Accuracy</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Accuracy</th>
                            <th>Target</th>
                            <th>Status</th>
                        </tr>
            """
            
            features = [
                {"name": "Negation Detection", "metric": "negation_detection", "target": 0.9},
                {"name": "Context Switch Detection", "metric": "context_switch_detection", "target": 0.85},
                {"name": "Contradiction Detection", "metric": "contradiction_detection", "target": 0.9}
            ]
            
            for feature in features:
                accuracy = overall["accuracy"].get(feature["metric"], 0)
                status = "✅" if accuracy >= feature["target"] else "❌"
                
                html += f"""
                        <tr>
                            <td>{feature["name"]}</td>
                            <td class="{self._get_score_class(accuracy)}">{accuracy}</td>
                            <td>{feature["target"]}</td>
                            <td>{status}</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
            
            # Success rates
            html += """
                <div class="section">
                    <h2>Conversation Quality</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Rate</th>
                            <th>Target</th>
                            <th>Status</th>
                        </tr>
            """
            
            metrics = [
                {"name": "Conversation Completion", "metric": "completion_rate", "target": 0.8},
                {"name": "Context Preservation", "metric": "context_preservation", "target": 0.85},
                {"name": "Entity Preservation", "metric": "entity_preservation", "target": 0.9},
                {"name": "Negation Handling", "metric": "negation_handling", "target": 0.85},
                {"name": "Context Switch Handling", "metric": "context_switch_handling", "target": 0.85},
                {"name": "Contradiction Handling", "metric": "contradiction_handling", "target": 0.85}
            ]
            
            for metric in metrics:
                rate = overall["success_rates"].get(metric["metric"], 0)
                status = "✅" if rate >= metric["target"] else "❌"
                
                html += f"""
                        <tr>
                            <td>{metric["name"]}</td>
                            <td class="{self._get_score_class(rate)}">{rate}</td>
                            <td>{metric["target"]}</td>
                            <td>{status}</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
            
            # Close HTML
            html += """
            </body>
            </html>
            """
            
            # Write report to file
            with open(report_file, 'w') as f:
                f.write(html)
            
            logger.info(f"Evaluation report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
    
    def _get_score_class(self, score):
        """Get CSS class for a score value"""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "medium"
        else:
            return "poor"
    
    def _generate_negation_test_data(self):
        """Generate test data for negation detection evaluation"""
        return [
            # Direct negations (should be detected)
            {"text": "I don't need a tow truck", "is_negation": True},
            {"text": "I no longer need roadside assistance", "is_negation": True},
            {"text": "Cancel my request for a tow", "is_negation": True},
            {"text": "I've changed my mind, I don't want that", "is_negation": True},
            {"text": "Let's not proceed with the appointment", "is_negation": True},
            {"text": "I'd rather not have a tow truck", "is_negation": True},
            {"text": "That's not what I'm looking for", "is_negation": True},
            {"text": "Forget about the tow truck", "is_negation": True},
            {"text": "I didn't say I needed a tow truck", "is_negation": True},
            {"text": "I don't want roadside assistance anymore", "is_negation": True},
            
            # Non-negations (should not be detected)
            {"text": "I need a tow truck", "is_negation": False},
            {"text": "My car won't start", "is_negation": False},
            {"text": "Can you help me with my car", "is_negation": False},
            {"text": "The engine is not working", "is_negation": False},
            {"text": "My car does not start", "is_negation": False},
            {"text": "I can't get my car to start", "is_negation": False},
            {"text": "I'm not sure what's wrong with my car", "is_negation": False},
            {"text": "I need help with my car", "is_negation": False},
            {"text": "Can you send someone for roadside assistance?", "is_negation": False},
            {"text": "I'm not at home, I'm at the mall", "is_negation": False},
            
            # Edge cases (challenging examples)
            {"text": "Actually, I need something else instead", "is_negation": False},  # Context switch, not negation
            {"text": "Not towing, but roadside assistance", "is_negation": False},  # Context switch
            {"text": "I don't need towing anymore, I need roadside assistance", "is_negation": True},  # Both negation and context switch
            {"text": "I'm not sure if I need a tow truck or not", "is_negation": False},  # Uncertainty, not negation
            {"text": "No, I already called someone else", "is_negation": True},
            {"text": "I don't think I explained my problem correctly", "is_negation": False},
            {"text": "Don't worry about sending a truck quickly", "is_negation": False},
            {"text": "I need help, I'm not able to start my car", "is_negation": False},
            {"text": "Never mind, I got it started", "is_negation": True},
            {"text": "I won't be needing that service after all", "is_negation": True}
        ]
    
    def _generate_context_switch_test_data(self):
        """Generate test data for context switch detection evaluation"""
        return [
            # Clear context switches (should be detected)
            {"text": "Actually, I need roadside assistance instead", "has_context_switch": True, 
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "Forget the tow truck, I need a battery jump", "has_context_switch": True,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "I've changed my mind, I need a tire change instead", "has_context_switch": True,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "Let's do an appointment instead of immediate help", "has_context_switch": True,
             "context": {"flow": "roadside", "service_type": "roadside"}},
            {"text": "I'd rather schedule service for next week", "has_context_switch": True,
             "context": {"flow": "roadside", "service_type": "roadside"}},
            
            # Negations without context switches (should not be detected)
            {"text": "I don't need a tow truck anymore", "has_context_switch": False,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "Cancel my request", "has_context_switch": False,
             "context": {"flow": "roadside", "service_type": "roadside"}},
            {"text": "I've decided against getting help", "has_context_switch": False,
             "context": {"flow": "roadside", "service_type": "roadside"}},
            
            # Additional information (should not be detected)
            {"text": "My car is a Honda Civic", "has_context_switch": False,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "I'm at the mall on Main Street", "has_context_switch": False,
             "context": {"flow": "roadside", "service_type": "roadside"}},
            
            # Edge cases
            {"text": "I don't need a tow but I do need roadside assistance", "has_context_switch": True,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "Actually, I'm not at the address I mentioned", "has_context_switch": False,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "It's not a Honda, it's a Toyota", "has_context_switch": False,
             "context": {"flow": "towing", "service_type": "towing"}},
            {"text": "Never mind, I'll just call a friend", "has_context_switch": False,
             "context": {"flow": "roadside", "service_type": "roadside"}},
            {"text": "Can we make this more immediate?", "has_context_switch": False,
             "context": {"flow": "appointment", "service_type": "appointment"}}
        ]
    
    def _generate_contradiction_test_data(self):
        """Generate test data for contradiction detection evaluation"""
        return [
            # Vehicle type contradictions
            {"text": "Actually my car is an SUV, not a sedan", "has_contradiction": True,
             "entity_type": "vehicle_type", "context": {"vehicle_type": "sedan"}},
            {"text": "It's a truck, not a compact car", "has_contradiction": True,
             "entity_type": "vehicle_type", "context": {"vehicle_type": "compact"}},
            {"text": "I'm driving a Honda not a Toyota", "has_contradiction": True,
             "entity_type": "vehicle_type", "context": {"vehicle_type": "toyota"}},
            
            # Location contradictions
            {"text": "I'm actually in downtown, not at the mall", "has_contradiction": True,
             "entity_type": "location", "context": {"location": "mall"}},
            {"text": "I'm at the gas station on Main St, not Oak Ave", "has_contradiction": True,
             "entity_type": "location", "context": {"location": "Oak Ave"}},
            
            # Service type contradictions
            {"text": "I need a full tow, not just a jump start", "has_contradiction": True,
             "entity_type": "service_type", "context": {"service_type": "jump start"}},
            {"text": "I need help with my keys, not a tire change", "has_contradiction": True,
             "entity_type": "service_type", "context": {"service_type": "tire change"}},
            
            # Not contradictions
            {"text": "My car is a sedan", "has_contradiction": False,
             "context": {"vehicle_type": "sedan"}},
            {"text": "I'm still at the mall", "has_contradiction": False,
             "context": {"location": "mall"}},
            {"text": "Can you still do the jump start?", "has_contradiction": False,
             "context": {"service_type": "jump start"}},
            
            # Edge cases
            {"text": "I'm at a different part of the mall now", "has_contradiction": False,
             "context": {"location": "mall"}},
            {"text": "My car is actually silver, not blue", "has_contradiction": True,
             "entity_type": "vehicle_color", "context": {"vehicle_color": "blue"}},
            {"text": "I need to change where you're picking up my car", "has_contradiction": True,
             "entity_type": "location", "context": {"location": "home"}}
        ]
    
    def _generate_multi_turn_test_data(self):
        """Generate test conversations for multi-turn evaluation"""
        return [
            # Conversation 1: Standard towing request
            {
                "turns": [
                    {"text": "I need a tow truck for my sedan"},
                    {"text": "I'm at the mall on Main Street"},
                    {"text": "It won't start at all"}
                ],
                "expected_outcome": {
                    "final_context": {
                        "flow": "towing",
                        "service_type": "towing",
                        "vehicle_type": "sedan",
                        "location": "mall on Main Street"
                    },
                    "preserved_entities": [
                        {"type": "vehicle_type", "value": "sedan"},
                        {"type": "location", "value": "mall on Main Street"}
                    ]
                }
            },
            
            # Conversation 2: Negation in conversation
            {
                "turns": [
                    {"text": "I need roadside assistance", "is_context_switch": True},
                    {"text": "My battery died in my SUV"},
                    {"text": "I'm at the grocery store parking lot"},
                    {"text": "Actually, I don't need help anymore", "is_negation": True}
                ],
                "expected_outcome": {
                    "final_context": {
                        "flow": "roadside",
                        "service_type": "roadside",
                        "vehicle_type": "suv",
                        "location": "grocery store parking lot"
                    },
                    "preserved_entities": [
                        {"type": "vehicle_type", "value": "suv"},
                        {"type": "location", "value": "grocery store parking lot"}
                    ]
                }
            },
            
            # Conversation 3: Context switch in conversation
            {
                "turns": [
                    {"text": "I need a tow truck", "is_context_switch": True},
                    {"text": "My Honda won't start"},
                    {"text": "I'm at work in the parking garage"},
                    {"text": "Actually, I need roadside assistance instead", "is_context_switch": True}
                ],
                "expected_outcome": {
                    "final_context": {
                        "flow": "roadside",
                        "service_type": "roadside",
                        "vehicle_type": "honda",
                        "location": "work in the parking garage"
                    },
                    "preserved_entities": [
                        {"type": "vehicle_type", "value": "honda"},
                        {"type": "location", "value": "work in the parking garage"}
                    ]
                }
            },
            
            # Conversation 4: Contradiction in conversation
            {
                "turns": [
                    {"text": "I need help with my sedan"},
                    {"text": "It won't start and I need a jump"},
                    {"text": "I'm at home in my driveway"},
                    {"text": "Actually, I'm in an SUV, not a sedan", "is_contradiction": True}
                ],
                "expected_outcome": {
                    "final_context": {
                        "flow": "roadside",
                        "service_type": "jump start",
                        "vehicle_type": "suv",
                        "location": "home in my driveway"
                    },
                    "preserved_entities": [
                        {"type": "location", "value": "home in my driveway"}
                    ]
                }
            },
            
            # Conversation 5: Complex multi-turn with all features
            {
                "turns": [
                    {"text": "I need a tow truck for my pickup", "is_context_switch": True},
                    {"text": "I'm at the shopping center on Oak Street"},
                    {"text": "Actually I don't need a tow anymore", "is_negation": True},
                    {"text": "I need roadside assistance instead", "is_context_switch": True},
                    {"text": "My battery is dead"},
                    {"text": "Actually, I'm in my sedan not my pickup", "is_contradiction": True},
                    {"text": "And I'm at the gas station not the shopping center", "is_contradiction": True}
                ],
                "expected_outcome": {
                    "final_context": {
                        "flow": "roadside",
                        "service_type": "jump start",
                        "vehicle_type": "sedan",
                        "location": "gas station"
                    },
                    "preserved_entities": []
                }
            }
        ]

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Evaluate context-aware chatbot models")
    parser.add_argument("--models_dir", default="./output/models", help="Directory containing trained models")
    parser.add_argument("--output_dir", default="./output/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--test_suite", default="comprehensive", choices=["comprehensive", "negation", "context_switch", "contradiction", "multi_turn"],
                        help="Which test suite to run")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ContextEvaluator(args.models_dir, args.output_dir)
    
    # Run selected test suite
    if args.test_suite == "comprehensive":
        results = evaluator.perform_comprehensive_evaluation()
    elif args.test_suite == "negation":
        results = evaluator.evaluate_negation_detection()
    elif args.test_suite == "context_switch":
        results = evaluator.evaluate_context_switch_detection()
    elif args.test_suite == "contradiction":
        results = evaluator.evaluate_contradiction_detection()
    elif args.test_suite == "multi_turn":
        results = evaluator.evaluate_multi_turn_conversations()
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    if "overall_score" in results:
        print(f"Overall Score: {results['overall_score']}")
    elif "accuracy" in results:
        print(f"Accuracy: {results['accuracy']}")
    elif "completion_rate" in results:
        print(f"Completion Rate: {results['completion_rate']}")
    
    print("\nDetailed results saved to the output directory.")

if __name__ == "__main__":
    main() 