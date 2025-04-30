# test_phase_dialog_1.py
import os
import sys
import unittest

# Ensure root directory is in path to find inference
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Mock NLUInferencer for testing DialogManager structure without full NLU
class MockNLUInferencer:
    def predict(self, text):
        print(f"MockNLUInferencer.predict called with: '{text}'")
        # Return a basic NLU structure based on keywords
        if "tow" in text.lower():
            return {
                "text": text,
                "intent": {"name": "towing_request_tow_basic", "confidence": 0.9},
                "entities": (
                    [{"entity": "service_type", "value": "tow"}]
                    if "tow" in text
                    else []
                ),
            }
        elif "battery" in text.lower():
            return {
                "text": text,
                "intent": {"name": "roadside_request_battery", "confidence": 0.85},
                "entities": [{"entity": "service_type", "value": "battery"}],
            }
        elif "appointment" in text.lower():
            return {
                "text": text,
                "intent": {
                    "name": "appointment_book_service_basic",
                    "confidence": 0.92,
                },
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
from dialog_manager import DialogManager, DialogState

# Restore original NLUInferencer if needed elsewhere (though not needed for this test)
inference.NLUInferencer = original_nlu_inferencer


class TestPhase1DialogSetup(unittest.TestCase):

    def test_dialog_state_init(self):
        """Test DialogState initialization."""
        print("\nTesting DialogState Initialization...")
        state = DialogState("test_id_1")
        self.assertEqual(state.conversation_id, "test_id_1")
        self.assertEqual(state.turn_count, 0)
        self.assertEqual(state.history, [])
        self.assertIsNone(state.current_intent)
        self.assertEqual(state.intent_confidence, 0.0)
        self.assertEqual(state.entities, {})
        self.assertEqual(state.required_slots, [])
        self.assertEqual(state.filled_slots, set())
        self.assertEqual(state.current_step, "START")
        self.assertIsNone(state.fallback_reason)
        print("DialogState Initialization Test PASSED.")

    def test_dialog_manager_init(self):
        """Test DialogManager initialization."""
        print("\nTesting DialogManager Initialization...")
        manager = DialogManager()
        self.assertIsNotNone(manager.nlu)  # Should be MockNLUInferencer instance
        self.assertIsInstance(manager.nlu, MockNLUInferencer)
        self.assertEqual(manager.states, {})
        print("DialogManager Initialization Test PASSED.")

    def test_get_or_create_state(self):
        """Test state retrieval and creation."""
        print("\nTesting State Creation/Retrieval...")
        manager = DialogManager()
        state1 = manager.get_or_create_state("conv1")
        self.assertIsInstance(state1, DialogState)
        self.assertEqual(state1.conversation_id, "conv1")

        state2 = manager.get_or_create_state("conv2")
        self.assertIsInstance(state2, DialogState)
        self.assertEqual(state2.conversation_id, "conv2")
        self.assertNotEqual(state1, state2)

        state1_retrieved = manager.get_or_create_state("conv1")
        self.assertEqual(state1, state1_retrieved)  # Should be the same object
        print("State Creation/Retrieval Test PASSED.")

    def test_state_update_from_nlu(self):
        """Test updating DialogState from mock NLU output."""
        print("\nTesting State Update from NLU...")
        state = DialogState("test_id_2")
        mock_nlu_result = {
            "text": "Need a tow for my Honda",
            "intent": {"name": "towing_request_tow_vehicle", "confidence": 0.88},
            "entities": [
                {"entity": "vehicle_make", "value": "Honda"},
                {"entity": "service_type", "value": "tow"},
            ],
        }
        state.update_from_nlu(mock_nlu_result)

        self.assertEqual(state.current_intent, "towing_request_tow_vehicle")
        self.assertEqual(state.intent_confidence, 0.88)
        self.assertEqual(
            state.entities, {"vehicle_make": "Honda", "service_type": "tow"}
        )
        self.assertEqual(state.filled_slots, {"vehicle_make", "service_type"})
        self.assertIsNone(state.fallback_reason)
        print("State Update from NLU Test PASSED.")

    def test_state_update_fallback(self):
        """Test updating DialogState for fallback cases."""
        print("\nTesting State Update for Fallback...")
        state = DialogState("test_id_3")
        mock_nlu_result = {
            "text": "asdsd",
            "intent": {"name": "fallback_low_confidence", "confidence": 0.2},
            "entities": [],
        }
        state.update_from_nlu(mock_nlu_result)
        self.assertEqual(state.fallback_reason, "low_confidence")

        state = DialogState("test_id_4")
        mock_nlu_result_2 = {
            "text": "weather?",
            "intent": {"name": "fallback_out_of_scope_weather", "confidence": 0.9},
            "entities": [],
        }
        state.update_from_nlu(mock_nlu_result_2)
        self.assertEqual(state.fallback_reason, "out_of_scope")
        print("State Update for Fallback Test PASSED.")

    def test_define_required_slots(self):
        """Test definition of required slots."""
        print("\nTesting Required Slot Definition...")
        manager = DialogManager()
        towing_slots = manager.define_required_slots("towing_request_tow_full")
        self.assertIn("pickup_location", towing_slots)
        self.assertIn("destination", towing_slots)
        self.assertIn("vehicle_make", towing_slots)

        roadside_slots = manager.define_required_slots("roadside_request_battery")
        self.assertIn("pickup_location", roadside_slots)
        self.assertNotIn("destination", roadside_slots)

        appointment_slots = manager.define_required_slots(
            "appointment_book_service_full"
        )
        self.assertIn("service_type", appointment_slots)
        self.assertIn("appointment_date", appointment_slots)

        fallback_slots = manager.define_required_slots("fallback_low_confidence")
        self.assertEqual(fallback_slots, [])
        print("Required Slot Definition Test PASSED.")

    def test_process_turn_basic_structure(self):
        """Test the basic structure of process_turn return value."""
        print("\nTesting process_turn Basic Structure...")
        manager = DialogManager()
        result = manager.process_turn("hello", "conv_test")
        self.assertIn("action", result)
        self.assertIn("state", result)
        self.assertIn("bot_response", result)
        self.assertIsInstance(result["state"], DialogState)
        self.assertIsInstance(result["action"], dict)
        self.assertIn("type", result["action"])
        print("process_turn Basic Structure Test PASSED.")


if __name__ == "__main__":
    print("--- Running Phase 1 Dialog Manager Tests ---")
    # Restore NLUInferencer before running tests if modified globally
    if "original_nlu_inferencer" in globals():
        inference.NLUInferencer = original_nlu_inferencer
    unittest.main()
    # Restore again after tests if needed
    if "original_nlu_inferencer" in globals():
        inference.NLUInferencer = original_nlu_inferencer
