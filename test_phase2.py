import os
import json
import unittest

class TestPhase2(unittest.TestCase):
    def test_model_directories_exist(self):
        """Test if the required model directories were created."""
        self.assertTrue(os.path.exists('./trained_nlu_model'), "Main model directory does not exist")
        self.assertTrue(os.path.exists('./trained_nlu_model/intent_model'), "Intent model directory does not exist")
        self.assertTrue(os.path.exists('./trained_nlu_model/entity_model'), "Entity model directory does not exist")
    
    def test_intent_mapping_file_exists(self):
        """Test if the intent mapping file exists."""
        intent_mapping_file = './trained_nlu_model/intent_model/intent2id.json'
        self.assertTrue(os.path.exists(intent_mapping_file), f"File does not exist: {intent_mapping_file}")
        
        # Check if the file contains valid JSON
        try:
            with open(intent_mapping_file, 'r') as f:
                intent_mappings = json.load(f)
                self.assertIsInstance(intent_mappings, dict, "Intent mappings should be a dictionary")
        except json.JSONDecodeError:
            self.fail("Intent mapping file contains invalid JSON")
    
    def test_entity_mapping_file_exists(self):
        """Test if the entity mapping file exists."""
        entity_mapping_file = './trained_nlu_model/entity_model/tag2id.json'
        self.assertTrue(os.path.exists(entity_mapping_file), f"File does not exist: {entity_mapping_file}")
        
        # Check if the file contains valid JSON
        try:
            with open(entity_mapping_file, 'r') as f:
                tag_mappings = json.load(f)
                self.assertIsInstance(tag_mappings, dict, "Tag mappings should be a dictionary")
                self.assertIn("O", tag_mappings, "Tag mappings should include 'O' (outside) tag")
        except json.JSONDecodeError:
            self.fail("Entity mapping file contains invalid JSON")

if __name__ == '__main__':
    unittest.main() 