# test_phase_dialog_2.py
import os
import sys
import unittest
from unittest.mock import patch

# Ensure root directory is in path to find inference
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Mock NLUInferencer for testing DialogManager logic
class MockNLUInferencer:
    def predict(self, text):
        print(f"MockNLUInferencer.predict called with: '{text}'")
        # Simulate NLU based on text content for testing slot filling
        text_lower = text.lower()
        intent_name = "unknown"
        confidence = 0.3  # Default low confidence
        entities = []

        # First check for location patterns that could apply to any intent
        if "123 main" in text_lower:
            entities.append({"entity": "pickup_location", "value": "123 Main St"})
            confidence = 0.9  # High confidence for specific location
        if "abc auto" in text_lower:
            entities.append({"entity": "destination", "value": "ABC Auto"})
            confidence = 0.9
        if "downtown" in text_lower or "shop" in text_lower:
            entities.append({"entity": "appointment_location", "value": "downtown"})
            confidence = 0.9

        # Check for date/time patterns
        if "tuesday" in text_lower:
            entities.append({"entity": "appointment_date", "value": "tuesday"})
            confidence = 0.9
        if "morning" in text_lower:
            entities.append({"entity": "appointment_time", "value": "morning"})
            confidence = 0.9

        # Then check for intent-specific patterns
        if "tow" in text_lower:
            intent_name = "towing_request_tow_location"
            confidence = 0.9
            if "honda" in text_lower:
                entities.append({"entity": "vehicle_make", "value": "Honda"})
            if "civic" in text_lower:
                entities.append({"entity": "vehicle_model", "value": "Civic"})
            if "2018" in text_lower:
                entities.append({"entity": "vehicle_year", "value": "2018"})

        elif "battery" in text_lower or "jump" in text_lower:
            intent_name = "roadside_request_battery"
            confidence = 0.88
            if "elm street" in text_lower:
                entities.append({"entity": "pickup_location", "value": "elm street"})

        elif (
            "appointment" in text_lower
            or "schedule" in text_lower
            or "oil change" in text_lower
        ):
            intent_name = "appointment_book_service_basic"
            confidence = 0.91
            if "oil change" in text_lower:
                entities.append({"entity": "service_type", "value": "oil change"})
            if "ford" in text_lower:
                entities.append({"entity": "vehicle_make", "value": "ford"})
                entities.append(
                    {"entity": "vehicle_model", "value": "F-150"}
                )  # Assume model
                entities.append(
                    {"entity": "vehicle_year", "value": "2020"}
                )  # Assume year

        # If we found entities but no clear intent, maintain high confidence and use "entity_only" intent
        if entities and intent_name == "unknown":
            intent_name = "entity_only"
            confidence = 0.85

        # Only trigger fallback if we have no entities and low confidence
        if not entities and confidence < 0.5:
            intent_name = "fallback_low_confidence"

        return {
            "text": text,
            "intent": {"name": intent_name, "confidence": confidence},
            "entities": entities,
        }


# Temporarily replace NLUInferencer during import
import inference

original_nlu_inferencer = inference.NLUInferencer
inference.NLUInferencer = MockNLUInferencer

# Now import DialogManager after mocking NLU
from dialog_manager import DialogManager, DialogState

# Restore original NLUInferencer if needed elsewhere
inference.NLUInferencer = original_nlu_inferencer


class TestPhase2DialogLogic(unittest.TestCase):

    def setUp(self):
        """Set up a new DialogManager for each test."""
        # We need to mock NLU for DialogManager tests
        self.patcher = patch("dialog_manager.NLUInferencer", MockNLUInferencer)
        self.MockNLU = self.patcher.start()
        self.manager = DialogManager()

    def tearDown(self):
        """Stop the patcher."""
        self.patcher.stop()

    def test_slot_filling_towing(self):
        """Test the happy path slot filling for towing."""
        print("\nTesting Towing Slot Filling...")
        conv_id = "tow_slots_1"

        # 1. Initial request
        result1 = self.manager.process_turn("I need a tow", conv_id)
        self.assertEqual(result1["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(
            result1["action"]["slot_name"], "pickup_location"
        )  # Expects pickup first
        self.assertEqual(result1["state"].current_step, "ASK_PICKUP_LOCATION")
        self.assertEqual(
            result1["state"].current_intent, "towing_request_tow_location"
        )  # NLU mock maps "tow" to this

        # 2. Provide pickup location
        result2 = self.manager.process_turn("I'm at 123 main st", conv_id)
        self.assertEqual(result2["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(
            result2["action"]["slot_name"], "destination"
        )  # Now asks for destination
        self.assertIn("pickup_location", result2["state"].filled_slots)
        self.assertEqual(result2["state"].entities["pickup_location"], "123 Main St")
        self.assertEqual(result2["state"].current_step, "ASK_DESTINATION")

        # 3. Provide destination
        result3 = self.manager.process_turn("to abc auto", conv_id)
        self.assertEqual(result3["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(
            result3["action"]["slot_name"], "vehicle_make"
        )  # Now asks for make
        self.assertIn("destination", result3["state"].filled_slots)
        self.assertEqual(
            result3["state"].entities["destination"], "ABC Auto"
        )  # Value from mock NLU
        self.assertEqual(result3["state"].current_step, "ASK_VEHICLE_MAKE")

        # 4. Provide make, model, year
        result4 = self.manager.process_turn("it's a 2018 honda civic", conv_id)
        self.assertEqual(
            result4["action"]["type"], "REQUEST_CONFIRMATION"
        )  # All slots filled
        self.assertIn("vehicle_make", result4["state"].filled_slots)
        self.assertIn("vehicle_model", result4["state"].filled_slots)
        self.assertIn("vehicle_year", result4["state"].filled_slots)
        self.assertEqual(result4["state"].current_step, "CONFIRMATION")
        self.assertIn("details", result4["action"])
        # Check details gathered
        self.assertEqual(result4["action"]["details"]["pickup_location"], "123 Main St")
        self.assertEqual(result4["action"]["details"]["destination"], "ABC Auto")
        self.assertEqual(result4["action"]["details"]["vehicle_make"], "Honda")

        # 5. Confirm
        result5 = self.manager.process_turn("yes confirm", conv_id)
        self.assertEqual(result5["action"]["type"], "RESPOND_COMPLETE")
        self.assertTrue(result5["state"].booking_confirmed)
        self.assertEqual(result5["state"].current_step, "COMPLETE")
        print("Towing Slot Filling Test PASSED.")

    def test_slot_filling_appointment(self):
        """Test the happy path slot filling for appointment."""
        print("\nTesting Appointment Slot Filling...")
        conv_id = "appt_slots_1"

        # 1. Initial request with some details
        result1 = self.manager.process_turn(
            "Need to schedule an oil change for my ford", conv_id
        )
        self.assertEqual(result1["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(
            result1["action"]["slot_name"], "appointment_location"
        )  # Ask for location next
        self.assertIn("service_type", result1["state"].filled_slots)
        self.assertIn(
            "vehicle_make", result1["state"].filled_slots
        )  # Mock adds model/year too
        self.assertIn("vehicle_model", result1["state"].filled_slots)
        self.assertIn("vehicle_year", result1["state"].filled_slots)
        self.assertEqual(result1["state"].current_step, "ASK_APPOINTMENT_LOCATION")

        # 2. Provide location
        result2 = self.manager.process_turn("at the downtown shop", conv_id)
        self.assertEqual(result2["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(
            result2["action"]["slot_name"], "appointment_date"
        )  # Ask for date
        self.assertIn("appointment_location", result2["state"].filled_slots)
        self.assertEqual(
            result2["state"].entities["appointment_location"], "downtown"
        )  # From mock NLU
        self.assertEqual(result2["state"].current_step, "ASK_APPOINTMENT_DATE")

        # 3. Provide date and time
        result3 = self.manager.process_turn("how about tuesday morning", conv_id)
        self.assertEqual(
            result3["action"]["type"], "REQUEST_CONFIRMATION"
        )  # All slots filled
        self.assertIn("appointment_date", result3["state"].filled_slots)
        self.assertIn("appointment_time", result3["state"].filled_slots)
        self.assertEqual(result3["state"].current_step, "CONFIRMATION")

        # 4. Confirm
        result4 = self.manager.process_turn("Sounds good", conv_id)
        self.assertEqual(result4["action"]["type"], "RESPOND_COMPLETE")
        self.assertTrue(result4["state"].booking_confirmed)
        print("Appointment Slot Filling Test PASSED.")

    def test_low_confidence_fallback(self):
        """Test that low confidence NLU triggers fallback action."""
        print("\nTesting Low Confidence Fallback...")
        conv_id = "fallback_1"
        result = self.manager.process_turn(
            "tell me about cars", conv_id
        )  # Mock NLU returns low confidence
        self.assertEqual(result["action"]["type"], "RESPOND_FALLBACK")
        self.assertEqual(result["action"]["reason"], "low_confidence")
        self.assertEqual(
            result["state"].current_step, "START"
        )  # Should reset if fallback
        print("Low Confidence Fallback Test PASSED.")


if __name__ == "__main__":
    print("--- Running Phase 2 Dialog Manager Tests ---")
    # Restore NLUInferencer before running tests if modified globally
    if "original_nlu_inferencer" in globals():
        inference.NLUInferencer = original_nlu_inferencer
    unittest.main()
    # Restore again after tests if needed
    if "original_nlu_inferencer" in globals():
        inference.NLUInferencer = original_nlu_inferencer
