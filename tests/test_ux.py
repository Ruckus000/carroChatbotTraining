"""
Test file for UI and UX components added in Phase 4
"""

import os
import sys
import unittest
import streamlit as st
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import get_performance_color
from utils.help_content import get_metric_help, get_section_help
from utils.export import export_to_csv, export_to_json

class TestUIComponents(unittest.TestCase):
    """
    Test UI components added in Phase 4
    """
    
    def test_get_performance_color(self):
        """Test the performance color function returns appropriate colors"""
        # Excellent performance (>=0.9)
        self.assertEqual(get_performance_color(0.95), "#4bff9d")
        self.assertEqual(get_performance_color(0.9), "#4bff9d")
        
        # Good performance (>=0.8)
        self.assertEqual(get_performance_color(0.85), "#a5f2a5")
        self.assertEqual(get_performance_color(0.8), "#a5f2a5")
        
        # Fair performance (>=0.7)
        self.assertEqual(get_performance_color(0.75), "#ffcb3c")
        self.assertEqual(get_performance_color(0.7), "#ffcb3c")
        
        # Poor performance (<0.7)
        self.assertEqual(get_performance_color(0.65), "#ff4b4b")
        self.assertEqual(get_performance_color(0.5), "#ff4b4b")
        self.assertEqual(get_performance_color(0), "#ff4b4b")

class TestHelpContent(unittest.TestCase):
    """
    Test help content used for tooltips and guides
    """
    
    def test_get_metric_help(self):
        """Test that metric help text is retrieved correctly"""
        self.assertIsInstance(get_metric_help("intent_f1"), str)
        self.assertIsInstance(get_metric_help("entity_precision"), str)
        self.assertIsInstance(get_metric_help("non_existent_metric"), str)  # Should return default message
    
    def test_get_section_help(self):
        """Test that section help text is retrieved correctly"""
        self.assertIsInstance(get_section_help("confusion_matrix"), str)
        self.assertIsInstance(get_section_help("error_patterns"), str)
        self.assertIsInstance(get_section_help("non_existent_section"), str)  # Should return default message

class TestExportFunctions(unittest.TestCase):
    """
    Test export functionality
    """
    
    def test_export_to_csv(self):
        """Test CSV export functionality"""
        test_metrics = {
            "intent_metrics": {
                "accuracy": 0.9,
                "precision": 0.85,
                "recall": 0.87,
                "f1": 0.86,
                "per_class_report": {
                    "intent_1": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 10},
                    "intent_2": {"precision": 0.7, "recall": 0.8, "f1-score": 0.75, "support": 20}
                }
            },
            "entity_metrics": {
                "micro avg": {"precision": 0.75, "recall": 0.8, "f1-score": 0.77, "support": 50},
                "macro avg": {"precision": 0.7, "recall": 0.75, "f1-score": 0.72, "support": 50},
                "entity_1": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75, "support": 30},
                "entity_2": {"precision": 0.6, "recall": 0.8, "f1-score": 0.7, "support": 20}
            }
        }
        
        csv_data, filename = export_to_csv(test_metrics, "test_model")
        
        # Check that data was created
        self.assertIsNotNone(csv_data)
        self.assertIsNotNone(filename)
        
        # Check filename format
        self.assertTrue(filename.startswith("nlu_metrics_test_model_"))
        self.assertTrue(filename.endswith(".csv"))
        
        # Check CSV content
        csv_content = csv_data.decode('utf-8')
        self.assertIn("INTENT METRICS", csv_content)
        self.assertIn("intent,intent_precision,intent_recall,intent_f1,intent_support", csv_content.lower())
        self.assertIn("intent_1", csv_content)
        self.assertIn("intent_2", csv_content)
    
    def test_export_to_json(self):
        """Test JSON export functionality"""
        test_metrics = {
            "intent_metrics": {
                "accuracy": 0.9,
                "f1": 0.85
            },
            "model_id": "test_model"
        }
        
        json_data, filename = export_to_json(test_metrics, "test_model")
        
        # Check that data was created
        self.assertIsNotNone(json_data)
        self.assertIsNotNone(filename)
        
        # Check filename format
        self.assertTrue(filename.startswith("nlu_metrics_test_model_"))
        self.assertTrue(filename.endswith(".json"))
        
        # Check JSON content
        import json
        json_content = json.loads(json_data.decode('utf-8'))
        self.assertEqual(json_content["intent_metrics"]["accuracy"], 0.9)
        self.assertEqual(json_content["intent_metrics"]["f1"], 0.85)
        self.assertEqual(json_content["model_id"], "test_model")
        self.assertIn("export_timestamp", json_content)

if __name__ == "__main__":
    unittest.main() 