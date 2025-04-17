#!/usr/bin/env python3
import os
import unittest
import json


class TestPhase5(unittest.TestCase):
    """Test that the project has been properly cleaned up, with only the required files remaining."""

    def test_required_files_exist(self):
        """Check that all required files exist."""
        required_files = [
            "train.py",
            "inference.py",
            "test_integration.py",
            "data/nlu_training_data.json",
            "requirements.txt",
            "README.md",
            ".gitignore",
            "test_phase1.py",
            "test_phase2.py",
            "test_phase3.py",
        ]

        for file_path in required_files:
            self.assertTrue(
                os.path.exists(file_path), f"Required file {file_path} does not exist"
            )

    def test_obsolete_files_dont_exist(self):
        """Check that obsolete files have been removed."""
        obsolete_files = [
            "chatbot_training.py",
            "model_training.py",
            "evaluation.py",
            "streamlit_app.py",
            "data/context_test_cases.json",
            "data/sample_conversations.json",
            "data/augmented_sample_conversations.json",
        ]

        for file_path in obsolete_files:
            self.assertFalse(
                os.path.exists(file_path), f"Obsolete file {file_path} still exists"
            )

    def test_obsolete_dirs_dont_exist(self):
        """Check that obsolete directories have been removed."""
        obsolete_dirs = ["langgraph_integration", "data/context_integration"]

        for dir_path in obsolete_dirs:
            self.assertFalse(
                os.path.exists(dir_path), f"Obsolete directory {dir_path} still exists"
            )

    def test_requirements_file_content(self):
        """Check that requirements.txt has the expected packages."""
        required_packages = [
            "transformers",
            "torch",
            "datasets",
            "scikit-learn",
            "numpy",
            "seqeval",
        ]

        with open("requirements.txt", "r") as f:
            content = f.read()

        for package in required_packages:
            self.assertIn(
                package, content, f"Required package {package} not in requirements.txt"
            )

    def test_data_integrity(self):
        """Check that nlu_training_data.json exists and has the expected structure."""
        self.assertTrue(os.path.exists("data/nlu_training_data.json"))

        try:
            with open("data/nlu_training_data.json", "r") as f:
                data = json.load(f)

            # Check that the data is a list
            self.assertIsInstance(data, list)

            # Check that it has at least one example
            self.assertGreater(len(data), 0)

            # Check structure of first example
            if len(data) > 0:
                example = data[0]
                self.assertIn("text", example)
                self.assertIn("intent", example)
                self.assertIn("entities", example)
        except Exception as e:
            self.fail(f"Error checking data/nlu_training_data.json: {e}")


if __name__ == "__main__":
    unittest.main()
