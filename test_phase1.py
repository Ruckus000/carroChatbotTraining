import json
import os
import unittest


class TestPhase1DataConsolidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate the consolidated data file if it doesn't exist"""
        if not os.path.exists("data/nlu_training_data.json"):
            os.system("python data_prep_phase1.py")

    def test_file_exists(self):
        self.assertTrue(
            os.path.exists("data/nlu_training_data.json"),
            "Consolidated data file not found",
        )

    def test_data_structure(self):
        with open("data/nlu_training_data.json", "r") as f:
            data = json.load(f)

        for example in data:
            self.assertIn("text", example)
            self.assertIn("intent", example)
            self.assertIn("entities", example)

            # Check intent format
            self.assertIn(
                "_", example["intent"], "Intent should be in flow_intent format"
            )

            # Check entities structure
            if example["entities"]:
                for entity in example["entities"]:
                    self.assertIn("entity", entity)
                    self.assertIn("value", entity)

    def test_fallback_example_exists(self):
        with open("data/nlu_training_data.json", "r") as f:
            data = json.load(f)

        fallbacks = [ex for ex in data if ex["intent"].startswith("fallback_")]
        self.assertGreaterEqual(len(fallbacks), 1, "No fallback examples found")


if __name__ == "__main__":
    unittest.main()
