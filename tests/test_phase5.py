import unittest
import os
import json
from pathlib import Path

class TestPhase5(unittest.TestCase):
    def test_project_structure(self):
        """Test that the project structure is correct and contains only the necessary files"""
        # Check required directories
        self.assertTrue(os.path.isdir("data"), "Data directory should exist")
        self.assertTrue(os.path.isdir("trained_nlu_model"), "Trained model directory should exist")
        
        # Check required core files
        required_files = [
            "train.py",
            "inference.py",
            "test_integration.py",
            "data/nlu_training_data.json",
            "requirements.txt",
            "README.md"
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.isfile(file_path), f"Required file {file_path} should exist")
        
        # Check the trained model contains the necessary files
        model_files = [
            "trained_nlu_model/intent_model/intent2id.json",
            "trained_nlu_model/entity_model/tag2id.json",
            "trained_nlu_model/intent_model/config.json",
            "trained_nlu_model/entity_model/config.json"
        ]
        
        for file_path in model_files:
            self.assertTrue(os.path.isfile(file_path), f"Model file {file_path} should exist")
        
        # Verify removed files are actually gone
        removed_files = [
            "langgraph_integration",
            "chatbot_training.py",
            "model_training.py",
            "context-integration.md",
            "train_context_models.py",
            "evaluation.py",
            "streamlit_app.py",
            "deploy.sh",
            "setup.py",
            "README-CONTEXT.md"
        ]
        
        for file_path in removed_files:
            if file_path.endswith("/") or "/" not in file_path and not file_path.endswith((".py", ".md", ".sh")):
                self.assertFalse(os.path.isdir(file_path), f"Directory {file_path} should be removed")
            else:
                self.assertFalse(os.path.isfile(file_path), f"File {file_path} should be removed")

    def test_requirements_file(self):
        """Test that requirements.txt contains only the necessary packages"""
        with open("requirements.txt", 'r') as f:
            requirements = f.read().splitlines()
        
        # Check essential packages are present
        essential_packages = [
            "transformers",
            "torch",
            "datasets",
            "scikit-learn",
            "numpy",
            "seqeval"
        ]
        
        for pkg in essential_packages:
            self.assertTrue(any(req.startswith(pkg) for req in requirements), f"Requirements should include {pkg}")
        
        # Check removed packages are gone
        removed_packages = [
            "langgraph",
            "langchain",
            "streamlit",
            "matplotlib",
            "seaborn"
        ]
        
        for pkg in removed_packages:
            self.assertFalse(any(req.startswith(pkg) for req in requirements), f"Requirements should not include {pkg}")

    def test_readme(self):
        """Test README contains the correct sections for the simplified system"""
        with open("README.md", 'r') as f:
            readme_content = f.read().lower()
        
        required_sections = [
            "overview",
            "setup",
            "training",
            "usage",
            "testing"
        ]
        
        for section in required_sections:
            self.assertIn(section, readme_content, f"README should include {section} section")
        
        # Check for training data format documentation
        self.assertIn("data/nlu_training_data.json", readme_content, "README should mention training data path")
        self.assertIn("intent", readme_content, "README should mention intent detection")
        self.assertIn("entities", readme_content, "README should mention entity detection")

    def test_training_script(self):
        """Test that the training script is compatible with the data format"""
        # Check if the training data exists and has the right format
        self.assertTrue(os.path.isfile("data/nlu_training_data.json"), "Training data file should exist")
        
        with open("data/nlu_training_data.json", 'r') as f:
            data = json.load(f)
        
        # Check data is a list
        self.assertIsInstance(data, list, "Training data should be a list of examples")
        
        # Check at least one example exists and has the right format
        self.assertGreater(len(data), 0, "Training data should contain examples")
        
        example = data[0]
        self.assertIn("text", example, "Example should have 'text' field")
        self.assertIn("intent", example, "Example should have 'intent' field")
        self.assertIn("entities", example, "Example should have 'entities' field")

if __name__ == "__main__":
    unittest.main() 