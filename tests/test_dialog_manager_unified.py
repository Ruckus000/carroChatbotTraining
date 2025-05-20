# /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/test_dialog_manager_unified.py
import unittest
import json
from typing import List, Dict, Any, Optional

# Mock NLU implementation for testing DialogManager
class MockNLUInferencer:
    def predict(self, text):
        print(f"DEBUG MOCK NLU: Predicting for text: '{text}'")
        # Basic mock - return low confidence by default
        base_result = {
            "text": text,
            "intent": {"name": "fallback_low_confidence", "confidence": 0.3},
            "entities": [],
            "sentiment": {"label": "neutral", "score": 0.5}  # Include sentiment data
        }

        # Specific test responses based on input text
        if "tow" in text.lower():
            base_result["intent"]["name"] = "towing_request_tow_basic"
            base_result["intent"]["confidence"] = 0.95
        elif "oil change" in text.lower():
            base_result["intent"]["name"] = "appointment_book_service_type"
            base_result["intent"]["confidence"] = 0.92
            base_result["entities"].append({"entity": "service_type", "value": "oil change"})
        elif "123 Main St" in text:
            base_result["intent"]["name"] = "entity_only"
            base_result["intent"]["confidence"] = 0.9
            base_result["entities"].append({"entity": "pickup_location", "value": "123 Main St"})
        elif "Honda" in text and "Civic" in text:
            base_result["intent"]["name"] = "entity_only"
            base_result["intent"]["confidence"] = 0.9
            base_result["entities"].append({"entity": "vehicle_make", "value": "Honda"})
            base_result["entities"].append({"entity": "vehicle_model", "value": "Civic"})
        elif "Reliable Auto" in text:
            base_result["intent"]["name"] = "entity_only"
            base_result["intent"]["confidence"] = 0.9
            base_result["entities"].append({"entity": "destination", "value": "Reliable Auto"})
        elif "tomorrow" in text.lower():
            base_result["intent"]["name"] = "entity_only"
            base_result["intent"]["confidence"] = 0.9
            base_result["entities"].append({"entity": "appointment_date", "value": "tomorrow"})
        elif "yes" in text.lower() or "confirm" in text.lower():
            base_result["intent"]["name"] = "affirm"
            base_result["intent"]["confidence"] = 0.95
        elif "no" in text.lower() or "cancel" in text.lower():
            base_result["intent"]["name"] = "deny"
            base_result["intent"]["confidence"] = 0.95

        # Add sentiment based on text content
        if any(word in text.lower() for word in ["happy", "great", "excellent", "good", "love"]):
            base_result["sentiment"]["label"] = "POSITIVE"
            base_result["sentiment"]["score"] = 0.9
        elif any(word in text.lower() for word in ["angry", "terrible", "bad", "hate", "awful"]):
            base_result["sentiment"]["label"] = "NEGATIVE"
            base_result["sentiment"]["score"] = 0.85

        return base_result

# Import the code under test - using mock NLU
from dialog_manager import DialogManager

class TestDialogManager(unittest.TestCase):

    def setUp(self):
        # Create a DialogManager instance with mock NLU
        self.nlu = MockNLUInferencer()
        self.dialog_manager = DialogManager(nlu_inferencer=self.nlu)

    def assertResponseRequests(self, response_text: str, possible_requests: List[str],
                            message: str = "Response should request specific information"):
        """Assert that the response is asking about at least one of the possible requests."""
        lowercase_text = response_text.lower()
        self.assertTrue(
            any(request in lowercase_text for request in possible_requests),
            f"{message}: '{response_text}' should contain one of {possible_requests}"
        )

    def assertResponseContainsAny(self, response_text: str, possible_contents: List[str],
                               message: str = "Response should contain specific information"):
        """Assert that the response contains at least one of the possible contents."""
        lowercase_text = response_text.lower()
        self.assertTrue(
            any(content in lowercase_text for content in possible_contents),
            f"{message}: '{response_text}' should contain one of {possible_contents}"
        )

    def assertStateHasEntity(self, state: Any, entity_type: str, value: Optional[str] = None):
        """Assert that the dialog state contains an entity of the specified type."""
        self.assertTrue(hasattr(state, 'entities'), "State should have 'entities' attribute")

        entity_exists = entity_type in state.entities
        self.assertTrue(entity_exists, f"State should contain entity of type '{entity_type}'")
        
        if value is not None and entity_exists:
            entity_value = state.entities[entity_type]
            self.assertEqual(value.lower(), entity_value.lower(),
                           f"Entity {entity_type} should have value '{value}'")

    def test_initialization(self):
        """Test the DialogManager initializes correctly."""
        self.assertIsNotNone(self.dialog_manager)
        # You can add more assertions about initial state if needed

    def test_towing_flow(self):
        """Test a complete towing conversation flow."""
        # Initial towing request
        conv_id = "test-towing-flow-123"
        response1 = self.dialog_manager.process_turn("I need a tow truck", conv_id)

        # Check the response asks for location
        self.assertResponseRequests(
            response1["bot_response"],
            ["location", "where", "address", "pickup", "pick up", "pickup location"],
            "First response should ask for pickup location"
        )

        # Provide location
        response2 = self.dialog_manager.process_turn("123 Main St", conv_id)

        # Check the response asks for destination
        self.assertResponseRequests(
            response2["bot_response"],
            ["destination", "where to", "take it", "drop off", "garage", "repair shop"],
            "Second response should ask for destination"
        )

        # Check the state contains the provided location
        self.assertStateHasEntity(response2["state"], "pickup_location", "123 Main St")

        # Provide destination
        response3 = self.dialog_manager.process_turn("Reliable Auto", conv_id)

        # Check the response asks for vehicle details
        self.assertResponseRequests(
            response3["bot_response"],
            ["vehicle", "car", "make", "model", "what kind of"],
            "Third response should ask for vehicle information"
        )

        # Check the state contains the provided destination
        self.assertStateHasEntity(response3["state"], "destination", "Reliable Auto")

        # Provide vehicle
        response4 = self.dialog_manager.process_turn("Honda Civic", conv_id)

        # Check the response asks for confirmation and includes collected information
        self.assertResponseRequests(
            response4["bot_response"],
            ["confirm", "correct", "right", "ok", "verification"],
            "Fourth response should ask for confirmation"
        )

        # Check the state contains the vehicle information
        self.assertStateHasEntity(response4["state"], "vehicle_make", "Honda")
        self.assertStateHasEntity(response4["state"], "vehicle_model", "Civic")

        # Confirm
        response5 = self.dialog_manager.process_turn("yes, confirm please", conv_id)

        # Check confirmation response
        self.assertResponseContainsAny(
            response5["bot_response"],
            ["confirm", "book", "schedule", "dispatch", "send", "thank", "ordered", "arranged"],
            "Final response should indicate successful booking"
        )

        # Check conversation history exists and has correct length
        self.assertEqual(response5["state"].turn_count, 5, "Should have 5 conversation turns")
        self.assertEqual(len(response5["state"].history), 5, "Should have 5 entries in history")

    def test_appointment_flow(self):
        """Test a basic appointment booking flow."""
        conv_id = "test-appointment-flow-456"

        # Initial appointment request
        response1 = self.dialog_manager.process_turn("I need an oil change", conv_id)

        # Check response asks for appointment details
        appointment_request_keywords = ["vehicle", "date", "time", "when", "location", "where", "service center"]
        self.assertResponseRequests(
            response1["bot_response"],
            appointment_request_keywords,
            "Response should ask for appointment details (vehicle, date, time, or location)"
        )

        # Provide a date
        response2 = self.dialog_manager.process_turn("tomorrow", conv_id)

        # Check response progresses the appointment flow
        appointment_progress_keywords = ["time", "confirm", "vehicle", "location", "service center", "when"]
        self.assertResponseRequests(
            response2["bot_response"],
            appointment_progress_keywords,
            "Response should progress appointment flow"
        )

        # Check entity was captured
        self.assertStateHasEntity(response2["state"], "appointment_date", "tomorrow")

    def test_separate_conversations(self):
        """Test that different conversation IDs maintain separate states."""
        conv_id1 = "test-conv-1"
        conv_id2 = "test-conv-2"

        # Start towing flow in first conversation
        response1 = self.dialog_manager.process_turn("I need a tow", conv_id1)

        # Start appointment flow in second conversation
        response2 = self.dialog_manager.process_turn("I need an oil change", conv_id2)

        # Continue first conversation
        response3 = self.dialog_manager.process_turn("123 Main St", conv_id1)

        # Assert both conversations maintained separate state - check intents
        self.assertTrue(
            "tow" in response1["state"].current_intent.lower(),
            f"First conversation should be in towing flow, got: {response1['state'].current_intent}"
        )
        self.assertTrue(
            "appointment" in response2["state"].current_intent.lower() or "service" in response2["state"].current_intent.lower(),
            f"Second conversation should be in appointment flow, got: {response2['state'].current_intent}"
        )

        # Check turn counts are correct
        self.assertEqual(response1["state"].turn_count, 1, "First conversation should have 1 turn initially")
        self.assertEqual(response3["state"].turn_count, 2, "First conversation should have 2 turns after continuation")
        self.assertEqual(response2["state"].turn_count, 1, "Second conversation should have 1 turn")

    def test_unknown_intent_handling(self):
        """Test how DialogManager handles unknown intents."""
        conv_id = "test-unknown-intent"
        response = self.dialog_manager.process_turn("What's the meaning of life?", conv_id)

        # Check response is a reasonable fallback/clarification
        fallback_keywords = [
            "i'm not sure", "don't understand", "could you", "help you with",
            "assist you with", "not clear", "can help", "try again", "rephrase"
        ]
        self.assertResponseContainsAny(
            response["bot_response"],
            fallback_keywords,
            "Response should handle unknown input gracefully"
        )

    def test_sentiment_processing(self):
        """Test that sentiment is correctly processed from NLU."""
        conv_id = "test-sentiment"

        # Test positive sentiment
        response = self.dialog_manager.process_turn("I am very happy with your service", conv_id)
        self.assertEqual(
            response["state"].current_sentiment["label"].upper(),
            "POSITIVE",
            "Should correctly identify positive sentiment"
        )

        # Test negative sentiment
        response = self.dialog_manager.process_turn("I am very angry about this situation", conv_id)
        self.assertEqual(
            response["state"].current_sentiment["label"].upper(),
            "NEGATIVE",
            "Should correctly identify negative sentiment"
        )

if __name__ == "__main__":
    unittest.main() 