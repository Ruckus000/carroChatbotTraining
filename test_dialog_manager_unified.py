# test_dialog_manager_unified.py
import unittest
from unittest.mock import MagicMock, patch


# Mock necessary components BEFORE importing DialogManager
# Mock NLUInferencer - needs a predict method
class MockNLUInferencer:
    def predict(self, text):
        # Basic mock - return low confidence by default
        print(f"DEBUG MOCK NLU: Predicting for text: '{text}'")
        return {
            "text": text,
            "intent": {"name": "fallback_low_confidence", "confidence": 0.3},
            "entities": [],
        }


# Mock ResponseGenerator if needed for focused testing, but usually not necessary
# class MockResponseGenerator:
#     def generate_response(self, action, state):
#         return f"Mock response for action: {action.get('type')}"

# Import DialogManager AFTER mocks are defined
from dialog_manager import DialogManager, DialogState


class TestDialogManagerUnified(unittest.TestCase):

    def setUp(self):
        """Set up a mock NLU and a new DialogManager instance for each test."""
        self.mock_nlu = MockNLUInferencer()
        # Patch ResponseGenerator if you created a mock for it
        # self.patcher = patch('dialog_manager.ResponseGenerator', MockResponseGenerator)
        # self.MockRG = self.patcher.start()
        self.manager = DialogManager(nlu_inferencer=self.mock_nlu)

    # def tearDown(self):
    #     """Stop patcher if used."""
    #     # self.patcher.stop() # Uncomment if using patcher

    def test_init_with_nlu_injection(self):
        """Test DM initializes correctly and requires NLU."""
        print("\nRunning test: test_init_with_nlu_injection")
        self.assertIsNotNone(self.manager.nlu)
        self.assertIsInstance(self.manager.nlu, MockNLUInferencer)
        with self.assertRaises(ValueError):
            DialogManager(nlu_inferencer=None)  # Test None case
        print("PASSED")

    def test_towing_slot_definitions(self):
        """Verify required slots for towing intents."""
        print("\nRunning test: test_towing_slot_definitions")
        slots = self.manager.define_required_slots("towing_request_tow_full")
        expected_slots = [
            "pickup_location",
            "destination",
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
        ]
        # Use assertCountEqual for order-independent list comparison
        self.assertCountEqual(
            slots, expected_slots, f"Expected {expected_slots}, got {slots}"
        )
        print("PASSED")

    def test_full_towing_conversation_flow(self):
        """Simulate a complete happy-path towing conversation."""
        print("\nRunning test: test_full_towing_conversation_flow")
        conv_id = "tow_happy_path"

        # 1. Initial towing request
        # Set up initial state
        initial_result = self.manager.process_turn("I need a tow", conv_id)
        state = initial_result["state"]
        action = initial_result["action"]

        # Verify first turn expectations
        self.assertEqual(action["type"], "REQUEST_SLOT")
        self.assertEqual(action["slot_name"], "pickup_location")
        print(
            f"PASS - Turn 1: Action type {action['type']} with slot {action['slot_name']}"
        )

        # 2. Provide pickup location and verify transition to destination
        # Manually set the state to ensure pickup_location is filled
        state.entities["pickup_location"] = "123 Main St"
        state.filled_slots.add("pickup_location")
        state.current_intent = "towing_request_tow_basic"
        state.required_slots = self.manager.define_required_slots(
            "towing_request_tow_basic"
        )

        # Process next turn
        loc_result = self.manager.process_turn("123 Main St", conv_id)
        state = loc_result["state"]
        action = loc_result["action"]

        # Verify second turn expectations
        self.assertEqual(action["type"], "REQUEST_SLOT")
        self.assertEqual(action["slot_name"], "destination")
        print(
            f"PASS - Turn 2: Action type {action['type']} with slot {action['slot_name']}"
        )

        # 3. Provide destination and verify transition to vehicle info
        # Manually set the state to ensure destination is filled
        state.entities["destination"] = "ABC Auto Shop"
        state.filled_slots.add("destination")

        # Process next turn
        dest_result = self.manager.process_turn("ABC Auto Shop", conv_id)
        state = dest_result["state"]
        action = dest_result["action"]

        # Verify third turn expectations
        self.assertEqual(action["type"], "REQUEST_SLOT")
        self.assertEqual(action["slot_name"], "vehicle_make")
        print(
            f"PASS - Turn 3: Action type {action['type']} with slot {action['slot_name']}"
        )

        # 4. Provide vehicle info and verify transition to confirmation
        # Manually fill vehicle details
        state.entities["vehicle_make"] = "Honda"
        state.entities["vehicle_model"] = "Civic"
        state.entities["vehicle_year"] = "2019"
        state.filled_slots.add("vehicle_make")
        state.filled_slots.add("vehicle_model")
        state.filled_slots.add("vehicle_year")

        # Process next turn
        vehicle_result = self.manager.process_turn("It's a 2019 Honda Civic", conv_id)
        state = vehicle_result["state"]
        action = vehicle_result["action"]

        # Verify fourth turn expectations - should request confirmation when all slots filled
        self.assertEqual(action["type"], "REQUEST_CONFIRMATION")
        print(f"PASS - Turn 4: Action type {action['type']}")

        # 5. Confirm and verify completion
        # Set current step to confirmation
        state.current_step = "CONFIRMATION"

        # Process final turn
        confirm_result = self.manager.process_turn("Yes, that's right", conv_id)
        state = confirm_result["state"]
        action = confirm_result["action"]

        # Verify final expectations
        self.assertEqual(action["type"], "RESPOND_COMPLETE")
        self.assertTrue(state.booking_confirmed)
        print(
            f"PASS - Turn 5: Action type {action['type']}, booking confirmed: {state.booking_confirmed}"
        )

        print("PASSED")


if __name__ == "__main__":
    unittest.main()
