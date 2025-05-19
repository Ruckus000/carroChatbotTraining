"""
Tests for visualization components in the NLU Benchmarking Dashboard
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization and data processing components
from utils.visualization import (
    create_intent_radar_chart,
    create_performance_timeline,
    create_confusion_matrix_heatmap,
    create_error_pattern_sankey,
    create_confidence_distribution
)

from utils.data_processing import (
    extract_intent_distributions,
    process_confusion_matrix,
    analyze_errors,
    process_history_data,
    extract_entity_metrics
)

class TestVisualizationComponents(unittest.TestCase):
    """Test suite for visualization components"""
    
    @patch('streamlit.warning')
    def test_create_intent_radar_chart(self, mock_warning):
        """Test creating an intent radar chart"""
        # Test with empty metrics
        result = create_intent_radar_chart({})
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        mock_warning.reset_mock()
        
        # Test with valid metrics
        metrics = {
            'intent_metrics': {
                'per_class_report': {
                    'intent1': {'f1-score': 0.8, 'precision': 0.7, 'recall': 0.9, 'support': 10},
                    'intent2': {'f1-score': 0.6, 'precision': 0.5, 'recall': 0.7, 'support': 15},
                    'intent3': {'f1-score': 0.9, 'precision': 0.9, 'recall': 0.9, 'support': 20},
                    'micro avg': {'f1-score': 0.8, 'precision': 0.7, 'recall': 0.9, 'support': 45}
                }
            }
        }
        
        result = create_intent_radar_chart(metrics)
        self.assertIsNotNone(result)
        
    @patch('streamlit.warning')
    def test_create_performance_timeline(self, mock_warning):
        """Test creating a performance timeline"""
        # Test with empty data
        df = pd.DataFrame()
        result = create_performance_timeline(df)
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        mock_warning.reset_mock()
        
        # Test with valid data
        data = {
            'timestamp': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'intent_f1': [0.8, 0.85, 0.9],
            'entity_f1': [0.7, 0.75, 0.8]
        }
        df = pd.DataFrame(data)
        
        result = create_performance_timeline(df)
        self.assertIsNotNone(result)
        
    @patch('streamlit.warning')
    def test_create_confusion_matrix_heatmap(self, mock_warning):
        """Test creating a confusion matrix heatmap"""
        # Test with empty data
        result = create_confusion_matrix_heatmap([], [])
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        mock_warning.reset_mock()
        
        # Test with valid data
        cm = [[10, 2, 1], [3, 15, 2], [1, 1, 20]]
        labels = ['intent1', 'intent2', 'intent3']
        
        result = create_confusion_matrix_heatmap(cm, labels)
        self.assertIsNotNone(result)
        
    @patch('streamlit.warning')
    def test_create_error_pattern_sankey(self, mock_warning):
        """Test creating an error pattern Sankey diagram"""
        # Test with empty data
        result = create_error_pattern_sankey([])
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        mock_warning.reset_mock()
        
        # Test with valid data
        errors = [
            {'true_intent': 'intent1', 'pred_intent': 'intent2', 'confidence': 0.7},
            {'true_intent': 'intent1', 'pred_intent': 'intent2', 'confidence': 0.8},
            {'true_intent': 'intent1', 'pred_intent': 'intent3', 'confidence': 0.6},
            {'true_intent': 'intent2', 'pred_intent': 'intent3', 'confidence': 0.9},
        ]
        
        result = create_error_pattern_sankey(errors)
        self.assertIsNotNone(result)
        
    @patch('streamlit.warning')
    def test_create_confidence_distribution(self, mock_warning):
        """Test creating a confidence distribution histogram"""
        # Test with empty data
        result = create_confidence_distribution([])
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        mock_warning.reset_mock()
        
        # Test with valid data
        errors = [
            {'confidence': 0.7},
            {'confidence': 0.8},
            {'confidence': 0.6},
            {'confidence': 0.9},
        ]
        
        correct = [
            {'confidence': 0.95},
            {'confidence': 0.92},
            {'confidence': 0.97},
        ]
        
        result = create_confidence_distribution(errors, correct)
        self.assertIsNotNone(result)

class TestDataProcessingFunctions(unittest.TestCase):
    """Test suite for data processing functions"""
    
    def test_extract_intent_distributions(self):
        """Test extracting intent distributions"""
        # Test with empty metrics
        result = extract_intent_distributions({})
        self.assertEqual(result, {})
        
        # Test with valid metrics
        metrics = {
            'intent_metrics': {
                'per_class_report': {
                    'intent1': {'f1-score': 0.8, 'precision': 0.7, 'recall': 0.9, 'support': 10},
                    'intent2': {'f1-score': 0.6, 'precision': 0.5, 'recall': 0.7, 'support': 15},
                    'intent3': {'f1-score': 0.9, 'precision': 0.9, 'recall': 0.9, 'support': 20},
                    'micro avg': {'f1-score': 0.8, 'precision': 0.7, 'recall': 0.9, 'support': 45}
                }
            }
        }
        
        result = extract_intent_distributions(metrics)
        self.assertIn('intent_data', result)
        # Total support should be the sum of all individual intent supports (10+15+20=45),
        # not including the micro avg support value
        self.assertEqual(result['total_support'], 45)
        self.assertEqual(len(result['intent_data']), 3)
        
    def test_process_confusion_matrix(self):
        """Test processing confusion matrix data"""
        # Test with empty data
        result = process_confusion_matrix([], [])
        self.assertEqual(result, {})
        
        # Test with valid data
        cm = [[10, 2, 1], [3, 15, 2], [1, 1, 20]]
        labels = ['intent1', 'intent2', 'intent3']
        
        result = process_confusion_matrix(cm, labels)
        self.assertIn('confusion_matrix', result)
        self.assertIn('normalized_matrix', result)
        self.assertIn('confused_pairs', result)
        self.assertEqual(len(result['confused_pairs']), 6)  # 6 non-diagonal elements
        
    def test_analyze_errors(self):
        """Test analyzing error patterns"""
        # Test with empty data
        result = analyze_errors([])
        self.assertEqual(result, {})
        
        # Test with valid data
        detailed_results = [
            {'text': 'example1', 'true_intent': 'intent1', 'pred_intent': 'intent2', 'confidence': 0.7, 'intent_correct': False},
            {'text': 'example2', 'true_intent': 'intent1', 'pred_intent': 'intent2', 'confidence': 0.8, 'intent_correct': False},
            {'text': 'example3', 'true_intent': 'intent1', 'pred_intent': 'intent3', 'confidence': 0.6, 'intent_correct': False},
            {'text': 'example4', 'true_intent': 'intent2', 'pred_intent': 'intent3', 'confidence': 0.9, 'intent_correct': False},
            {'text': 'example5', 'true_intent': 'intent1', 'pred_intent': 'intent1', 'confidence': 0.95, 'intent_correct': True},
            {'text': 'example6', 'true_intent': 'intent2', 'pred_intent': 'intent2', 'confidence': 0.92, 'intent_correct': True},
        ]
        
        result = analyze_errors(detailed_results)
        self.assertEqual(result['total_examples'], 6)
        self.assertEqual(result['error_count'], 4)
        self.assertAlmostEqual(result['error_rate'], 4/6)
        self.assertEqual(len(result['error_patterns']), 3)
        
    def test_process_history_data(self):
        """Test processing historical data"""
        # Test with empty data
        df = pd.DataFrame()
        result = process_history_data(df)
        self.assertEqual(result, {})
        
        # Test with valid data
        data = {
            'timestamp': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'intent_accuracy': [0.75, 0.8, 0.85, 0.9],
            'intent_f1': [0.72, 0.78, 0.83, 0.88],
            'entity_f1': [0.7, 0.6, 0.75, 0.85],
            'model_id': ['model_v1', 'model_v2', 'model_v3', 'model_v4']
        }
        df = pd.DataFrame(data)
        
        result = process_history_data(df)
        self.assertIn('history_df', result)
        self.assertIn('changes', result)
        self.assertIn('trends', result)
        self.assertIn('latest_metrics', result)
        
    def test_extract_entity_metrics(self):
        """Test extracting entity metrics"""
        # Test with empty metrics
        result = extract_entity_metrics({})
        self.assertEqual(result, {})
        
        # Test with valid metrics
        metrics = {
            'entity_metrics': {
                'report': {
                    'B-person': {'f1-score': 0.8, 'precision': 0.7, 'recall': 0.9, 'support': 10},
                    'B-location': {'f1-score': 0.6, 'precision': 0.5, 'recall': 0.7, 'support': 15},
                    'B-organization': {'f1-score': 0.9, 'precision': 0.9, 'recall': 0.9, 'support': 20},
                    'micro avg': {'f1-score': 0.8, 'precision': 0.7, 'recall': 0.9, 'support': 45}
                }
            }
        }
        
        result = extract_entity_metrics(metrics)
        self.assertIn('entity_data', result)
        self.assertIn('micro avg', result)
        self.assertEqual(len(result['entity_data']), 3)

if __name__ == '__main__':
    unittest.main() 