import os
import json
import unittest


class TestPhase2(unittest.TestCase):
    def test_model_directories_exist(self):
        """Test if the required model directories were created."""
        self.assertTrue(
            os.path.exists("./trained_nlu_model"), "Main model directory does not exist"
        )
        self.assertTrue(
            os.path.exists("./trained_nlu_model/intent_model"),
            "Intent model directory does not exist",
        )
        self.assertTrue(
            os.path.exists("./trained_nlu_model/entity_model"),
            "Entity model directory does not exist",
        )

    def test_intent_model_files_exist(self):
        """Test if all required intent model files exist."""
        intent_files = [
            "./trained_nlu_model/intent_model/config.json",
            "./trained_nlu_model/intent_model/tokenizer_config.json",
            "./trained_nlu_model/intent_model/vocab.txt",
            "./trained_nlu_model/intent_model/intent2id.json",
            "./trained_nlu_model/intent_model/special_tokens_map.json",
        ]

        # Check for either pytorch_model.bin or model.safetensors
        self.assertTrue(
            os.path.exists("./trained_nlu_model/intent_model/pytorch_model.bin")
            or os.path.exists("./trained_nlu_model/intent_model/model.safetensors"),
            "Neither pytorch_model.bin nor model.safetensors exists in intent model directory",
        )

        for file_path in intent_files:
            self.assertTrue(
                os.path.exists(file_path), f"File does not exist: {file_path}"
            )

    def test_entity_model_files_exist(self):
        """Test if all required entity model files exist."""
        entity_files = [
            "./trained_nlu_model/entity_model/config.json",
            "./trained_nlu_model/entity_model/tokenizer_config.json",
            "./trained_nlu_model/entity_model/vocab.txt",
            "./trained_nlu_model/entity_model/tag2id.json",
            "./trained_nlu_model/entity_model/special_tokens_map.json",
        ]

        # Check for either pytorch_model.bin or model.safetensors
        self.assertTrue(
            os.path.exists("./trained_nlu_model/entity_model/pytorch_model.bin")
            or os.path.exists("./trained_nlu_model/entity_model/model.safetensors"),
            "Neither pytorch_model.bin nor model.safetensors exists in entity model directory",
        )

        for file_path in entity_files:
            self.assertTrue(
                os.path.exists(file_path), f"File does not exist: {file_path}"
            )

    def test_intent_mappings(self):
        """Test if intent mappings were created correctly."""
        try:
            with open("./trained_nlu_model/intent_model/intent2id.json", "r") as f:
                intent2id = json.load(f)

            # Check that mappings exist and are not empty
            self.assertTrue(intent2id, "Intent mappings are empty")

            # Check format (intent string -> integer ID)
            for intent, id_val in intent2id.items():
                self.assertIsInstance(
                    intent, str, f"Intent key is not a string: {intent}"
                )
                self.assertIsInstance(
                    id_val, int, f"Intent ID is not an integer: {id_val}"
                )

            # Check that we have reasonable number of intents (at least a few)
            self.assertGreater(len(intent2id), 3, "Too few intents in mappings")

            # Check that some expected intents exist
            for prefix in ["towing", "roadside", "appointment", "fallback"]:
                self.assertTrue(
                    any(intent.startswith(prefix) for intent in intent2id.keys()),
                    f"No intent with prefix '{prefix}' found",
                )

        except FileNotFoundError:
            self.fail("Intent mappings file not found")
        except json.JSONDecodeError:
            self.fail("Intent mappings file contains invalid JSON")

    def test_entity_mappings(self):
        """Test if entity mappings were created correctly."""
        try:
            with open("./trained_nlu_model/entity_model/tag2id.json", "r") as f:
                tag2id = json.load(f)

            # Check that mappings exist and are not empty
            self.assertTrue(tag2id, "Entity tag mappings are empty")

            # Check format (BIO tag string -> integer ID)
            for tag, id_val in tag2id.items():
                self.assertIsInstance(tag, str, f"Tag key is not a string: {tag}")
                self.assertIsInstance(
                    id_val, int, f"Tag ID is not an integer: {id_val}"
                )

            # Check that 'O' tag exists and is mapped to 0
            self.assertIn("O", tag2id, "Outside tag 'O' missing from mappings")
            self.assertEqual(tag2id["O"], 0, "Outside tag 'O' should be mapped to ID 0")

            # Check BIO format (tags should be O, B-*, or I-*)
            for tag in tag2id.keys():
                if tag != "O":
                    self.assertTrue(
                        tag.startswith("B-") or tag.startswith("I-"),
                        f"Entity tag '{tag}' does not follow BIO format",
                    )

            # Check that we have at least a few entity types
            entity_types = set()
            for tag in tag2id.keys():
                if tag.startswith("B-") or tag.startswith("I-"):
                    entity_types.add(tag[2:])  # Extract entity type

            self.assertGreater(len(entity_types), 1, "Too few entity types in mappings")

            # Check that some expected entity types exist (or similar ones)
            common_entities = ["location", "time", "date", "vehicle"]
            self.assertTrue(
                any(
                    any(entity in entity_type for entity in common_entities)
                    for entity_type in entity_types
                ),
                "No common entity types found",
            )

        except FileNotFoundError:
            self.fail("Entity mappings file not found")
        except json.JSONDecodeError:
            self.fail("Entity mappings file contains invalid JSON")

    def test_model_configs(self):
        """Test if model config files contain the expected information."""
        # Test intent model config
        try:
            with open("./trained_nlu_model/intent_model/config.json", "r") as f:
                config = json.load(f)

            self.assertIn(
                "id2label", config, "Intent model config missing id2label mapping"
            )
            self.assertIn(
                "label2id", config, "Intent model config missing label2id mapping"
            )

        except FileNotFoundError:
            self.fail("Intent model config file not found")
        except json.JSONDecodeError:
            self.fail("Intent model config file contains invalid JSON")

        # Test entity model config
        try:
            with open("./trained_nlu_model/entity_model/config.json", "r") as f:
                config = json.load(f)

            self.assertIn(
                "id2label", config, "Entity model config missing id2label mapping"
            )
            self.assertIn(
                "label2id", config, "Entity model config missing label2id mapping"
            )

        except FileNotFoundError:
            self.fail("Entity model config file not found")
        except json.JSONDecodeError:
            self.fail("Entity model config file contains invalid JSON")


if __name__ == "__main__":
    unittest.main()
