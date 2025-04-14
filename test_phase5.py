import os
import unittest

class TestPhase5(unittest.TestCase):
    def test_required_files_exist(self):
        """Test that required files exist."""
        required_files = [
            'README.md',
            'requirements.txt'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Required file {file_path} does not exist")
    
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
            'scikit-learn'
        ]
        
        for package in essential_packages:
            self.assertIn(package, requirements, f"requirements.txt is missing {package}")

if __name__ == '__main__':
    unittest.main() 