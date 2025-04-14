import os
import unittest

class TestPhase5(unittest.TestCase):
    def test_required_files_exist(self):
        """Test that all required files exist."""
        required_files = [
            # Core implementation files
            'train.py',
            'inference.py',
            
            # Test files
            'test_phase1.py',
            'test_phase2.py',
            'test_phase3.py',
            'test_integration.py',
            'test_phase5.py',
            
            # Documentation and configuration
            'README.md',
            'requirements.txt',
            
            # Data file
            'data/nlu_training_data.json'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Required file {file_path} does not exist")
    
    def test_model_directories_exist(self):
        """Test that model directories exist."""
        required_dirs = [
            './trained_nlu_model',
            './trained_nlu_model/intent_model',
            './trained_nlu_model/entity_model'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(os.path.isdir(dir_path), f"Required directory {dir_path} does not exist")
    
    def test_readme_content(self):
        """Test if README.md contains required sections."""
        with open('README.md', 'r') as f:
            readme_content = f.read().lower()
        
        required_sections = [
            'installation',
            'data format',
            'training',
            'inference',
            'testing'
        ]
        
        for section in required_sections:
            self.assertIn(section, readme_content, f"README is missing section about '{section}'")
    
    def test_requirements_content(self):
        """Test if requirements.txt contains essential packages."""
        with open('requirements.txt', 'r') as f:
            requirements = f.read().lower()
        
        essential_packages = [
            'transformers',
            'torch',
            'numpy',
            'scikit-learn',
            'seqeval',
            'datasets'
        ]
        
        for package in essential_packages:
            self.assertIn(package, requirements, f"requirements.txt is missing {package}")
    
    def test_data_file_structure(self):
        """Test if the data file has the correct structure."""
        import json
        
        try:
            with open('data/nlu_training_data.json', 'r') as f:
                data = json.load(f)
            
            self.assertTrue(isinstance(data, list), "Data file should contain a list of examples")
            
            if len(data) > 0:
                example = data[0]
                self.assertIn('text', example, "Examples should have a 'text' field")
                self.assertIn('intent', example, "Examples should have an 'intent' field")
                self.assertIn('entities', example, "Examples should have an 'entities' field")
                self.assertTrue(isinstance(example['entities'], list), "Entities field should be a list")
        except FileNotFoundError:
            self.fail("Data file not found")
        except json.JSONDecodeError:
            self.fail("Data file contains invalid JSON")

if __name__ == '__main__':
    unittest.main() 