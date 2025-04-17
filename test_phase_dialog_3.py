# test_phase_dialog_3.py
import unittest
import os
import sys
from unittest.mock import patch

# Ensure root directory is in path to find inference and dialog_manager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Mock NLUInferencer for testing DialogManager structure without full NLU
class MockNLUInferencer:
    # Define predict to return different results based on input for testing various paths
    def predict(self, text):
        text_lower = text.lower()
        if (
            "tow" in text_lower
            and "123 main" in text_lower
            and "abc auto" in text_lower
            and "honda civic 2018" in text_lower
        ):
            return {
                "text": text,
                "intent": {"name": "towing_request_tow_full", "confidence": 0.99},
                "entities": [
                    {"entity": "pickup_location", "value": "123 Main St"},
                    {"entity": "destination", "value": "ABC Auto"},
                    {"entity": "vehicle_make", "value": "Honda"},
                    {"entity": "vehicle_model", "value": "Civic"},
                    {"entity": "vehicle_year", "value": "2018"},
                ],
            }
        elif "tow" in text_lower:
            return {
                "text": text,
                "intent": {"name": "towing_request_tow_basic", "confidence": 0.9},
                "entities": [],
            }
        elif "battery" in text_lower:
            return {
                "text": text,
                "intent": {"name": "roadside_request_battery", "confidence": 0.85},
                "entities": [],
            }
        elif "appointment" in text_lower:
            return {
                "text": text,
                "intent": {
                    "name": "appointment_book_service_basic",
                    "confidence": 0.92,
                },
                "entities": [],
            }
        elif (
            "yes" in text_lower or "confirm" in text_lower or "that's all" in text_lower
        ):  # Simulate confirmation
            return {
                "text": text,
                "intent": {"name": "affirm", "confidence": 0.99},
                "entities": [],
            }
        else:
            return {
                "text": text,
                "intent": {"name": "fallback_low_confidence", "confidence": 0.3},
                "entities": [],
            }


# Temporarily replace NLUInferencer during import
import inference

original_nlu_inferencer = inference.NLUInferencer
inference.NLUInferencer = MockNLUInferencer

# Now import DialogManager after mocking NLU
from dialog_manager import DialogState, DialogManager
from response_generator import ResponseGenerator  # Import ResponseGenerator

# Restore original NLUInferencer if needed elsewhere
inference.NLUInferencer = original_nlu_inferencer


class TestPhase3ResponseGeneration(unittest.TestCase):

    def setUp(self):
        """Set up a new DialogManager for each test."""
        # We need to mock NLU for DialogManager tests
        self.patcher = patch("dialog_manager.NLUInferencer", MockNLUInferencer)
        self.MockNLU = self.patcher.start()
        self.manager = DialogManager()
        # Ensure the manager uses the actual ResponseGenerator
        self.manager.response_generator = ResponseGenerator()

    def tearDown(self):
        """Stop the patcher."""
        self.patcher.stop()

    def test_response_for_request_slot(self):
        """Test response generation when asking for a slot."""
        print("\nTesting Response Generation (Request Slot)...")
        conv_id = "resp_slot_1"
        result = self.manager.process_turn("I need a tow", conv_id)
        self.assertEqual(result["action"]["type"], "REQUEST_SLOT")
        # Using more flexible assertions that match any of our pickup_location templates
        self.assertTrue(
            any(
                phrase in result["bot_response"].lower()
                for phrase in ["vehicle", "where", "pickup", "located", "tow truck"]
            )
        )
        print(f"Response: {result['bot_response']}")
        print("Response Generation (Request Slot) Test PASSED.")

    def test_response_for_confirmation(self):
        """Test response generation when requesting confirmation."""
        print("\nTesting Response Generation (Confirmation)...")
        conv_id = "resp_confirm_1"
        # Simulate filling all slots for towing
        state = self.manager.get_or_create_state(conv_id)
        state.current_intent = "towing_request_tow_full"
        state.entities = {
            "pickup_location": "456 Oak Ave",
            "destination": "City Repair",
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "vehicle_year": "2020",
        }
        state.required_slots = [
            "pickup_location",
            "destination",
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
        ]
        state.filled_slots = set(state.required_slots)
        state.current_step = "CONFIRMATION"  # Set to CONFIRMATION directly

        # Now directly create a confirmation action
        action = {"type": "REQUEST_CONFIRMATION", "details": state.entities}
        response = self.manager.response_generator.generate_response(action, state)

        self.assertIn("toyota", response.lower())
        self.assertIn("camry", response.lower())
        self.assertIn("oak", response.lower())
        print(f"Response: {response}")
        print("Response Generation (Confirmation) Test PASSED.")

    def test_response_for_completion(self):
        """Test response generation upon task completion."""
        print("\nTesting Response Generation (Completion)...")
        conv_id = "resp_complete_1"
        # Simulate being at the confirmation step
        state = self.manager.get_or_create_state(conv_id)
        state.current_intent = "towing_request_tow_full"
        state.entities = {
            "pickup_location": "Work",
            "destination": "Home",
            "vehicle_make": "Kia",
            "vehicle_model": "Soul",
            "vehicle_year": "2021",
        }
        state.required_slots = list(state.entities.keys())
        state.filled_slots = set(state.required_slots)
        state.current_step = "CONFIRMATION"

        # Create completion action directly
        action = {
            "type": "RESPOND_COMPLETE",
            "details": state.entities,
            "intent": state.current_intent,
        }
        response = self.manager.response_generator.generate_response(action, state)

        self.assertTrue(
            any(
                word in response.lower()
                for word in [
                    "confirmed",
                    "booked",
                    "dispatched",
                    "en route",
                    "set",
                    "help",
                ]
            )
        )
        self.assertIn("kia", response.lower())
        print(f"Response: {response}")
        print("Response Generation (Completion) Test PASSED.")

    def test_response_for_fallback(self):
        """Test response generation for fallback scenarios."""
        print("\nTesting Response Generation (Fallback)...")
        conv_id = "resp_fallback_1"
        # Create fallback action directly
        state = self.manager.get_or_create_state(conv_id)
        action = {"type": "RESPOND_FALLBACK", "reason": "low_confidence"}
        response = self.manager.response_generator.generate_response(action, state)

        # Check for any of the possible fallback phrases from our template options
        self.assertTrue(
            any(
                phrase in response.lower()
                for phrase in [
                    "understand",
                    "clarity",
                    "rephrase",
                    "didn't",
                    "sorry",
                    "got that",
                    "more detail",
                ]
            )
        )
        print(f"Response: {response}")
        print("Response Generation (Fallback) Test PASSED.")


if __name__ == "__main__":
    print("--- Running Phase 3 Dialog Manager Tests ---")
    # Restore NLUInferencer before running tests if modified globally
    if "original_nlu_inferencer" in globals():
        inference.NLUInferencer = original_nlu_inferencer
    unittest.main()
    # Restore again after tests if needed
    if "original_nlu_inferencer" in globals():
        inference.NLUInferencer = original_nlu_inferencer
