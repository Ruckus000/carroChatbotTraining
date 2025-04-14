import json
import os
import unittest

class TestPhase1DataConsolidation(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(os.path.exists('data/nlu_training_data.json'),
                        "Consolidated data file not found")

    def test_data_structure(self):
        with open('data/nlu_training_data.json', 'r') as f:
            data = json.load(f)
            
        for example in data:
            self.assertIn('text', example)
            self.assertIn('intent', example)
            self.assertIn('entities', example)
            
            # Check text is a string
            self.assertIsInstance(example['text'], str)
            
            # Check intent is a string with underscore
            self.assertIsInstance(example['intent'], str)
            self.assertIn('_', example['intent'], 
                        "Intent should be in format with underscore (e.g., flow_intent)")
            
            # Check entities structure
            self.assertIsInstance(example['entities'], list)
            if example['entities']:
                for entity in example['entities']:
                    self.assertIn('entity', entity)
                    self.assertIn('value', entity)
                    self.assertIsInstance(entity['entity'], str)
                    self.assertIsInstance(entity['value'], str)

    def test_fallback_example_exists(self):
        with open('data/nlu_training_data.json', 'r') as f:
            data = json.load(f)
            
        fallbacks = [ex for ex in data 
                   if ex['intent'].startswith('fallback_')]
        self.assertGreaterEqual(len(fallbacks), 1,
                              "No fallback examples found")

if __name__ == '__main__':
    unittest.main() 