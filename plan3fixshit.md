This plan is designed with extreme detail and redundancy checks, assuming Cursor might misinterpret or skip steps. Each phase includes explicit instructions and verification steps _within_ the phase itself to force adherence. We will build the Dialog Manager and Response Generator incrementally.

**Overarching Goal:** Implement a functional Dialog Manager and Response Generator that uses the NLU output from `inference.py` to guide conversations for Towing, Roadside Assistance, and Appointments according to the user's vision document, culminating in simulated booking/confirmation messages.

**Prerequisites:**

- Phase 1-4 of the previous simplification plan (Path A) are fully complete.
- `train.py` has successfully run and created the `./trained_nlu_model/` directory with valid models.
- `inference.py` contains _only_ the `NLUInferencer` class.
- The root `test_integration.py` passes using the real models.
- The project structure is clean, containing primarily `data/nlu_training_data.json`, `train.py`, `inference.py`, `requirements.txt`, `README.md`, and the test files.

**General Instructions for Cursor (Reinforced):**

- **Execute Sequentially & Verify:** Complete ALL steps within a phase, including running the specified verification commands/checks, before proceeding. DO NOT skip verification steps.
- **Targeted Modifications:** Only modify or create the files explicitly mentioned in each step. DO NOT modify `train.py`, `inference.py`, or any test scripts (`test_*.py`).
- **Use Provided Code Snippets:** When code snippets are provided, use them exactly unless explicitly told to adapt them.
- **Error Reporting:** If any command fails, a test fails after you've modified the code for that phase, or instructions are unclear, STOP and report the exact step, the command/code used, and the full error message.

---

**Phase 1: Setup and Basic Dialog State Tracking**

**Goal:** Create the basic structure for the Dialog Manager, define the conversation state, and implement the initial state update based on NLU output.

**Steps:**

1.  **Create `dialog_manager.py`:** Create a new file named `dialog_manager.py` in the project root.
2.  **Define `DialogState` Class:** Inside `dialog_manager.py`, create a class named `DialogState` to hold conversation information.

    ```python
    # dialog_manager.py
    import copy

    class DialogState:
        def __init__(self, conversation_id=None):
            self.conversation_id = conversation_id or "default_session"
            self.turn_count = 0
            self.history = [] # List of {"user": text, "bot": text}
            self.current_intent = None
            self.intent_confidence = 0.0
            self.entities = {} # {entity_type: value} e.g. {"pickup_location": "123 Main St"}
            self.required_slots = [] # List of entity types needed for current intent
            self.filled_slots = set() # Set of entity types already filled
            self.current_step = "START" # Tracks progress within a flow
            self.fallback_reason = None # 'low_confidence', 'out_of_scope', 'ambiguous'
            self.booking_details = {} # Stores final booking info
            self.booking_confirmed = False

        def update_from_nlu(self, nlu_result):
            """Update state based on NLU output."""
            self.current_intent = nlu_result.get("intent", {}).get("name", "unknown")
            self.intent_confidence = nlu_result.get("intent", {}).get("confidence", 0.0)

            # Update entities, preferring new values
            nlu_entities = nlu_result.get("entities", [])
            for entity_info in nlu_entities:
                entity_type = entity_info.get("entity")
                entity_value = entity_info.get("value")
                if entity_type and entity_value:
                    self.entities[entity_type] = entity_value
                    self.filled_slots.add(entity_type) # Mark as filled

            # Check for fallback/clarification from NLU
            if self.current_intent == "fallback_low_confidence":
                self.fallback_reason = "low_confidence"
            elif self.current_intent.startswith("fallback_"):
                 self.fallback_reason = "out_of_scope"
            elif self.current_intent.startswith("clarification_"):
                 self.fallback_reason = "ambiguous"
            else:
                 self.fallback_reason = None

        def add_history(self, user_input, bot_response):
             """Add interaction to history."""
             self.history.append({"user": user_input, "bot": bot_response})
             self.turn_count += 1

        def get_missing_slots(self):
            """Return a list of required slots that are not yet filled."""
            return [slot for slot in self.required_slots if slot not in self.filled_slots]

        def reset_flow(self):
            """Reset state specific to the current flow/intent."""
            print(f"DEBUG: Resetting flow. Current state: {self.current_intent}, {self.current_step}")
            self.current_intent = None
            self.intent_confidence = 0.0
            self.entities = {}
            self.required_slots = []
            self.filled_slots = set()
            self.current_step = "START"
            self.fallback_reason = None
            self.booking_details = {}
            self.booking_confirmed = False
            print(f"DEBUG: State after reset: {self.current_intent}, {self.current_step}")

    # --- End of DialogState Class ---
    ```

3.  **Define `DialogManager` Class:** In the same `dialog_manager.py` file, below the `DialogState` class, create a class named `DialogManager`.

    ```python
    # dialog_manager.py (continued)
    from inference import NLUInferencer # Import the NLU class

    class DialogManager:
        def __init__(self):
            try:
                self.nlu = NLUInferencer() # Load NLU models
                print("NLUInferencer loaded successfully.")
            except Exception as e:
                print(f"FATAL ERROR: Could not initialize NLUInferencer: {e}")
                # In a real app, might exit or disable NLU features
                self.nlu = None # Indicate NLU failure

            self.states = {} # Dictionary to store states per conversation_id

        def get_or_create_state(self, conversation_id):
            """Get existing state or create a new one."""
            if conversation_id not in self.states:
                print(f"Creating new state for conversation_id: {conversation_id}")
                self.states[conversation_id] = DialogState(conversation_id)
            return self.states[conversation_id]

        def define_required_slots(self, intent):
             """Define required entities (slots) for each intent."""
             # Based on user vision doc
             if intent.startswith("towing_"):
                 return ["pickup_location", "destination", "vehicle_make", "vehicle_model", "vehicle_year"]
             elif intent == "roadside_request_battery":
                 return ["pickup_location"]
             elif intent == "roadside_request_tire":
                 return ["pickup_location", "vehicle_make", "vehicle_model"] # Need vehicle for tire equip
             elif intent in ["roadside_request_fuel", "roadside_request_keys"]:
                 return ["pickup_location"]
             elif intent.startswith("roadside_"): # General roadside
                 return ["service_type", "pickup_location"] # Ask for specific service first
             elif intent.startswith("appointment_"):
                 return ["service_type", "appointment_location", "vehicle_make", "vehicle_model", "vehicle_year", "appointment_date", "appointment_time"]
             else:
                 return [] # No slots needed for fallback/clarification/basic intents initially

        def process_turn(self, user_input, conversation_id=None):
            """Process one turn of the conversation."""
            if not self.nlu:
                return {"response": "Error: NLU system not available.", "state": None}

            state = self.get_or_create_state(conversation_id)
            state.turn_count += 1 # Increment turn count

            # 1. Get NLU Result
            try:
                nlu_result = self.nlu.predict(user_input)
                print(f"DEBUG NLU Result: {nlu_result}")
            except Exception as e:
                print(f"ERROR during NLU prediction: {e}")
                state.fallback_reason = "nlu_error"
                nlu_result = {"intent": {"name": "fallback_nlu_error", "confidence": 1.0}, "entities": []}

            # 2. Update State with NLU result
            state.update_from_nlu(nlu_result)

            # If the NLU immediately detected fallback/clarification, handle it
            if state.fallback_reason:
                action = {"type": "RESPOND_FALLBACK", "reason": state.fallback_reason}
            # If NLU gives a valid intent AND we are at the START, set required slots
            elif state.current_intent and state.current_intent != "unknown" and not state.current_intent.startswith("fallback") and not state.current_intent.startswith("clarification") and state.current_step == "START":
                state.required_slots = self.define_required_slots(state.current_intent)
                state.filled_slots.intersection_update(state.entities.keys()) # Keep only current entities
                print(f"DEBUG: Intent '{state.current_intent}' detected. Required slots: {state.required_slots}. Already filled: {state.filled_slots}")
                missing_slots = state.get_missing_slots()
                if not missing_slots:
                     state.current_step = "CONFIRMATION"
                     action = {"type": "REQUEST_CONFIRMATION", "details": state.entities}
                else:
                     next_slot = missing_slots[0]
                     state.current_step = f"ASK_{next_slot.upper()}"
                     action = {"type": "REQUEST_SLOT", "slot_name": next_slot}
            # If we are already in a flow (asking for slots)
            elif state.current_step.startswith("ASK_"):
                 # Check if the *last requested slot* is now filled by NLU entities
                 last_asked_slot = state.current_step.replace("ASK_", "").lower()
                 if last_asked_slot in state.filled_slots:
                     print(f"DEBUG: Slot '{last_asked_slot}' filled.")
                     missing_slots = state.get_missing_slots()
                     if not missing_slots:
                         state.current_step = "CONFIRMATION"
                         action = {"type": "REQUEST_CONFIRMATION", "details": state.entities}
                     else:
                         next_slot = missing_slots[0]
                         state.current_step = f"ASK_{next_slot.upper()}"
                         action = {"type": "REQUEST_SLOT", "slot_name": next_slot}
                 else:
                     # Slot not filled, re-ask for the same slot
                     print(f"DEBUG: Slot '{last_asked_slot}' still missing. Re-asking.")
                     action = {"type": "REQUEST_SLOT", "slot_name": last_asked_slot}
            # Handle confirmation step
            elif state.current_step == "CONFIRMATION":
                 # For now, assume confirmation is positive if intent is not clearly 'no' or 'change'
                 # More robust confirmation handling needed later
                 if "no" in user_input.lower() or "change" in user_input.lower() or "wrong" in user_input.lower():
                     # Simple reset for now if user says no/change
                     action = {"type": "RESPOND_RESTART_FLOW", "reason": "User declined confirmation."}
                     state.reset_flow() # Reset flow state
                 else:
                     # Assume confirmed
                     state.booking_confirmed = True
                     state.booking_details = state.entities # Store confirmed details
                     state.current_step = "COMPLETE"
                     action = {"type": "RESPOND_COMPLETE", "details": state.booking_details, "intent": state.current_intent}
            # Default: If intent is unknown or no specific step matches, clarify
            else:
                 state.fallback_reason = "clarification_needed"
                 action = {"type": "RESPOND_FALLBACK", "reason": state.fallback_reason}

            print(f"DEBUG Action decided: {action}")
            # We will add response generation later
            bot_response = f"Action: {action['type']}" # Placeholder response

            # Update history
            state.add_history(user_input, bot_response)

            # Return the chosen action and the updated state
            return {"action": action, "state": state, "bot_response": bot_response}

    # --- End of DialogManager Class ---
    ```

4.  **Basic Imports:** Add `import copy` at the top of `dialog_manager.py`. Also add `from inference import NLUInferencer`.
5.  **Save File:** Save `dialog_manager.py`.

**Phase 1 Test:**

- **Instruction:** Create a new file named `test_phase_dialog_1.py`. Paste the following code into it. **DO NOT MODIFY THIS TEST SCRIPT.** Run it using `python test_phase_dialog_1.py`.
- **Debugging:** If the test fails:
  - Check for syntax errors in `dialog_manager.py`.
  - Ensure the `DialogState` and `DialogManager` classes are defined correctly.
  - Verify the `NLUInferencer` import works (this assumes Phase 1-3 of the previous plan completed).
  - Ensure the `get_or_create_state`, `update_from_nlu`, `define_required_slots`, `get_missing_slots`, and `add_history` methods exist within their respective classes and handle basic cases without crashing.
  - Fix `dialog_manager.py` and repeat until the test passes. Report success or failure.

```python
# test_phase_dialog_1.py
import unittest
import os
import sys

# Ensure root directory is in path to find inference
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock NLUInferencer for testing DialogManager structure without full NLU
class MockNLUInferencer:
    def predict(self, text):
        print(f"MockNLUInferencer.predict called with: '{text}'")
        # Return a basic NLU structure based on keywords
        if "tow" in text.lower():
            return {
                "text": text,
                "intent": {"name": "towing_request_tow_basic", "confidence": 0.9},
                "entities": [{"entity": "service_type", "value": "tow"}] if "tow" in text else []
            }
        elif "battery" in text.lower():
             return {
                "text": text,
                "intent": {"name": "roadside_request_battery", "confidence": 0.85},
                "entities": [{"entity": "service_type", "value": "battery"}]
            }
        elif "appointment" in text.lower():
             return {
                "text": text,
                "intent": {"name": "appointment_book_service_basic", "confidence": 0.92},
                "entities": []
            }
        else:
            return {
                "text": text,
                "intent": {"name": "fallback_low_confidence", "confidence": 0.3},
                "entities": []
            }

# Temporarily replace NLUInferencer during import
import inference
original_nlu_inferencer = inference.NLUInferencer
inference.NLUInferencer = MockNLUInferencer

# Now import DialogManager after mocking NLU
from dialog_manager import DialogState, DialogManager

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
        self.assertIsNotNone(manager.nlu) # Should be MockNLUInferencer instance
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
        self.assertEqual(state1, state1_retrieved) # Should be the same object
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
                {"entity": "service_type", "value": "tow"}
            ]
        }
        state.update_from_nlu(mock_nlu_result)

        self.assertEqual(state.current_intent, "towing_request_tow_vehicle")
        self.assertEqual(state.intent_confidence, 0.88)
        self.assertEqual(state.entities, {"vehicle_make": "Honda", "service_type": "tow"})
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
             "entities": []
        }
        state.update_from_nlu(mock_nlu_result)
        self.assertEqual(state.fallback_reason, "low_confidence")

        state = DialogState("test_id_4")
        mock_nlu_result_2 = {
             "text": "weather?",
             "intent": {"name": "fallback_out_of_scope_weather", "confidence": 0.9},
             "entities": []
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

        appointment_slots = manager.define_required_slots("appointment_book_service_full")
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

if __name__ == '__main__':
    print("--- Running Phase 1 Dialog Manager Tests ---")
    # Restore NLUInferencer before running tests if modified globally
    if 'original_nlu_inferencer' in globals():
        inference.NLUInferencer = original_nlu_inferencer
    unittest.main()
    # Restore again after tests if needed
    if 'original_nlu_inferencer' in globals():
        inference.NLUInferencer = original_nlu_inferencer
```

---

**Phase 2: Implement Core Dialog Logic (Slot Filling)**

**Goal:** Enhance the `DialogManager.process_turn` method to handle the "happy path" slot filling based on the defined required slots for each intent.

**Steps:**

1.  **Locate `dialog_manager.py`:** Ensure the file exists.
2.  **Edit `DialogManager.process_turn`:**

    - Find the main logic block within `process_turn` (after NLU prediction and state update).
    - **Replace** the existing placeholder logic block (starting around `if state.fallback_reason:` and ending before `state.add_history(...)`) with the following more detailed logic:

    ```python
            # ---------------------------------------------------
            # START OF REPLACEMENT LOGIC BLOCK
            # ---------------------------------------------------
            action = None # Initialize action

            # 3. Determine Next Action based on State

            # Handle Fallback/Clarification first
            if state.fallback_reason:
                print(f"DEBUG: Handling fallback/clarification. Reason: {state.fallback_reason}")
                action = {"type": "RESPOND_FALLBACK", "reason": state.fallback_reason}
                # Reset flow if we fell back, unless it was just low confidence on a known intent
                if state.fallback_reason != "low_confidence":
                     state.reset_flow()

            # Handle Confirmation Step
            elif state.current_step == "CONFIRMATION":
                 print(f"DEBUG: Handling confirmation step.")
                 # Simple Yes/No check for now
                 if "no" in user_input.lower() or "change" in user_input.lower() or "wrong" in user_input.lower():
                     action = {"type": "RESPOND_RESTART_FLOW", "reason": "User declined confirmation."}
                     print(f"DEBUG: User declined confirmation. Resetting flow.")
                     state.reset_flow() # Simple reset for beginner implementation
                 else:
                     # Assume confirmed
                     state.booking_confirmed = True
                     state.booking_details = copy.deepcopy(state.entities) # Store confirmed details
                     state.current_step = "COMPLETE"
                     print(f"DEBUG: Booking Confirmed. Details: {state.booking_details}")
                     action = {"type": "RESPOND_COMPLETE", "details": state.booking_details, "intent": state.current_intent}

            # Handle task completion
            elif state.current_step == "COMPLETE":
                print(f"DEBUG: Task already complete. Offering further help.")
                # If the task is already complete, maybe just offer help again or end.
                action = {"type": "RESPOND_ALREADY_COMPLETE"}
                state.reset_flow() # Reset for next interaction

            # Handle START or asking for slots
            else:
                # Check if we have a valid, non-fallback/clarification intent
                is_valid_intent = state.current_intent and \
                                  state.current_intent != "unknown" and \
                                  not state.current_intent.startswith("fallback") and \
                                  not state.current_intent.startswith("clarification")

                if is_valid_intent:
                    # If we just detected a new valid intent (state was START), set required slots
                    if state.current_step == "START":
                        state.required_slots = self.define_required_slots(state.current_intent)
                        # Recalculate filled slots based ONLY on current NLU result for the first turn of an intent
                        current_nlu_entities = {entity_info["entity"] for entity_info in nlu_result.get("entities", [])}
                        state.filled_slots = current_nlu_entities
                        print(f"DEBUG: New intent '{state.current_intent}'. Required: {state.required_slots}. Filled now: {state.filled_slots}")
                    # Else (we are already in progress asking for slots):
                    # state.filled_slots was already updated in state.update_from_nlu

                    # Find missing slots
                    missing_slots = state.get_missing_slots()
                    print(f"DEBUG: Checking missing slots. Required: {state.required_slots}. Filled: {state.filled_slots}. Missing: {missing_slots}")

                    if not missing_slots:
                         # All slots filled, move to confirmation
                         state.current_step = "CONFIRMATION"
                         print(f"DEBUG: All slots filled. Moving to CONFIRMATION.")
                         action = {"type": "REQUEST_CONFIRMATION", "details": copy.deepcopy(state.entities)}
                    else:
                         # Ask for the next missing slot
                         next_slot_to_ask = missing_slots[0]
                         state.current_step = f"ASK_{next_slot_to_ask.upper()}"
                         print(f"DEBUG: Asking for next slot: {next_slot_to_ask}")
                         action = {"type": "REQUEST_SLOT", "slot_name": next_slot_to_ask}
                else:
                    # No valid intent detected, and not already handled as fallback/clarification
                    print(f"DEBUG: No valid intent or required slots defined. Clarifying.")
                    state.fallback_reason = "clarification_needed"
                    action = {"type": "RESPOND_FALLBACK", "reason": state.fallback_reason}

            # Ensure an action was decided
            if action is None:
                 print("ERROR: No action decided! Defaulting to fallback.")
                 state.fallback_reason = "internal_error"
                 action = {"type": "RESPOND_FALLBACK", "reason": state.fallback_reason}

            # ---------------------------------------------------
            # END OF REPLACEMENT LOGIC BLOCK
            # ---------------------------------------------------
    ```

    - Ensure this new block replaces the previous logic between the NLU update and the history update.
    - Add `import copy` at the top of `dialog_manager.py`.

3.  **Save File:** Save the changes to `dialog_manager.py`.

**Phase 2 Test:**

- **Instruction:** Create a new file named `test_phase_dialog_2.py`. Paste the following code into it. **DO NOT MODIFY THIS TEST SCRIPT.** Run it using `python test_phase_dialog_2.py`.
- **Debugging:** If the test fails:
  - Carefully compare the logic flow in your `DialogManager.process_turn` with the provided code block, paying close attention to the conditions (`if/elif/else`).
  - Check the `DEBUG` print statements in the console output when running the test to trace the state transitions (`current_step`, `required_slots`, `filled_slots`, `action`).
  - Verify the `define_required_slots` method returns the correct lists.
  - Ensure the `DialogState.update_from_nlu` method correctly updates `filled_slots`.
  - Fix `dialog_manager.py`. Repeat until `test_phase_dialog_2.py` passes. Report success or failure.

```python
# test_phase_dialog_2.py
import unittest
import os
import sys

# Ensure root directory is in path to find inference
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock NLUInferencer for testing DialogManager logic
class MockNLUInferencer:
    def predict(self, text):
        print(f"MockNLUInferencer.predict called with: '{text}'")
        # Simulate NLU based on text content for testing slot filling
        text_lower = text.lower()
        intent_name = "unknown"
        confidence = 0.3 # Default low confidence
        entities = []

        if "tow" in text_lower:
            intent_name = "towing_request_tow_location" # Assume full intent if tow is mentioned
            confidence = 0.9
            if "123 main" in text_lower:
                entities.append({"entity": "pickup_location", "value": "123 Main St"})
            if "abc auto" in text_lower:
                entities.append({"entity": "destination", "value": "ABC Auto"})
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

        elif "appointment" in text_lower or "schedule" in text_lower or "oil change" in text_lower:
             intent_name = "appointment_book_service_basic"
             confidence = 0.91
             if "oil change" in text_lower:
                 entities.append({"entity": "service_type", "value": "oil change"})
             if "downtown" in text_lower:
                 entities.append({"entity": "appointment_location", "value": "downtown"})
             if "ford" in text_lower:
                 entities.append({"entity": "vehicle_make", "value": "ford"})
                 entities.append({"entity": "vehicle_model", "value": "F-150"}) # Assume model
                 entities.append({"entity": "vehicle_year", "value": "2020"}) # Assume year
             if "tuesday" in text_lower:
                 entities.append({"entity": "appointment_date", "value": "tuesday"})
             if "morning" in text_lower:
                 entities.append({"entity": "appointment_time", "value": "morning"})

        # Simple fallback trigger
        if confidence < 0.5 and intent_name == "unknown":
            intent_name = "fallback_low_confidence"

        return {
            "text": text,
            "intent": {"name": intent_name, "confidence": confidence},
            "entities": entities
        }


# Temporarily replace NLUInferencer during import
import inference
original_nlu_inferencer = inference.NLUInferencer
inference.NLUInferencer = MockNLUInferencer

# Now import DialogManager after mocking NLU
from dialog_manager import DialogState, DialogManager

# Restore original NLUInferencer if needed elsewhere
inference.NLUInferencer = original_nlu_inferencer

class TestPhase2DialogLogic(unittest.TestCase):

    def setUp(self):
        """Set up a new DialogManager for each test."""
        # We need to mock NLU for DialogManager tests
        self.patcher = patch('dialog_manager.NLUInferencer', MockNLUInferencer)
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
        self.assertEqual(result1["action"]["slot_name"], "pickup_location") # Expects pickup first
        self.assertEqual(result1["state"].current_step, "ASK_PICKUP_LOCATION")
        self.assertEqual(result1["state"].current_intent, "towing_request_tow_location") # NLU mock maps "tow" to this

        # 2. Provide pickup location
        result2 = self.manager.process_turn("I'm at 123 main st", conv_id)
        self.assertEqual(result2["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(result2["action"]["slot_name"], "destination") # Now asks for destination
        self.assertIn("pickup_location", result2["state"].filled_slots)
        self.assertEqual(result2["state"].entities["pickup_location"], "123 Main St")
        self.assertEqual(result2["state"].current_step, "ASK_DESTINATION")

        # 3. Provide destination
        result3 = self.manager.process_turn("to abc auto", conv_id)
        self.assertEqual(result3["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(result3["action"]["slot_name"], "vehicle_make") # Now asks for make
        self.assertIn("destination", result3["state"].filled_slots)
        self.assertEqual(result3["state"].entities["destination"], "ABC Auto") # Value from mock NLU
        self.assertEqual(result3["state"].current_step, "ASK_VEHICLE_MAKE")

        # 4. Provide make, model, year
        result4 = self.manager.process_turn("it's a 2018 honda civic", conv_id)
        self.assertEqual(result4["action"]["type"], "REQUEST_CONFIRMATION") # All slots filled
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
        result1 = self.manager.process_turn("Need to schedule an oil change for my ford", conv_id)
        self.assertEqual(result1["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(result1["action"]["slot_name"], "appointment_location") # Ask for location next
        self.assertIn("service_type", result1["state"].filled_slots)
        self.assertIn("vehicle_make", result1["state"].filled_slots) # Mock adds model/year too
        self.assertIn("vehicle_model", result1["state"].filled_slots)
        self.assertIn("vehicle_year", result1["state"].filled_slots)
        self.assertEqual(result1["state"].current_step, "ASK_APPOINTMENT_LOCATION")

        # 2. Provide location
        result2 = self.manager.process_turn("at the downtown shop", conv_id)
        self.assertEqual(result2["action"]["type"], "REQUEST_SLOT")
        self.assertEqual(result2["action"]["slot_name"], "appointment_date") # Ask for date
        self.assertIn("appointment_location", result2["state"].filled_slots)
        self.assertEqual(result2["state"].entities["appointment_location"], "downtown") # From mock NLU
        self.assertEqual(result2["state"].current_step, "ASK_APPOINTMENT_DATE")

        # 3. Provide date and time
        result3 = self.manager.process_turn("how about tuesday morning", conv_id)
        self.assertEqual(result3["action"]["type"], "REQUEST_CONFIRMATION") # All slots filled
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
        result = self.manager.process_turn("tell me about cars", conv_id) # Mock NLU returns low confidence
        self.assertEqual(result["action"]["type"], "RESPOND_FALLBACK")
        self.assertEqual(result["action"]["reason"], "low_confidence")
        self.assertEqual(result["state"].current_step, "START") # Should reset if fallback
        print("Low Confidence Fallback Test PASSED.")


if __name__ == '__main__':
    print("--- Running Phase 2 Dialog Manager Tests ---")
    # Restore NLUInferencer before running tests if modified globally
    if 'original_nlu_inferencer' in globals():
        inference.NLUInferencer = original_nlu_inferencer
    unittest.main()
    # Restore again after tests if needed
    if 'original_nlu_inferencer' in globals():
        inference.NLUInferencer = original_nlu_inferencer
```

---

**Phase 3: Implement Response Generation**

**Goal:** Create a `ResponseGenerator` class and integrate it into the `DialogManager` to produce user-facing text based on the action decided by the dialog logic.

**Steps:**

1.  **Create `response_generator.py`:** Create a new file named `response_generator.py`.
2.  **Define `ResponseGenerator` Class:** Inside `response_generator.py`, create the class.

    ```python
    # response_generator.py
    import random

    class ResponseGenerator:
        def __init__(self):
            # Define response templates based on actions and context
            self.templates = {
                "REQUEST_SLOT": {
                    "pickup_location": [
                        "Okay, I can help with that. Where is your vehicle located?",
                        "Got it. Where should the tow truck pick up your vehicle?",
                        "To proceed, please provide the pickup location for your vehicle."
                    ],
                    "destination": [
                        "Thanks! And where does the vehicle need to be towed to?",
                        "What's the destination address or shop name?",
                        "Please provide the drop-off location."
                    ],
                    "vehicle_make": [
                        "Okay, I have the locations. What is the make of your vehicle?",
                        "Can you tell me the make of the car?",
                        "What vehicle make should I note down?"
                    ],
                    "vehicle_model": [
                        "Thanks. And the model?",
                        "What model is the {vehicle_make}?", # Example using context
                        "Please provide the vehicle model."
                    ],
                    "vehicle_year": [
                        "Almost done with vehicle details. What year is the {vehicle_make} {vehicle_model}?",
                        "What's the year of the vehicle?",
                        "Please tell me the vehicle year."
                    ],
                    "service_type": [ # For general roadside or appointment start
                        "I can help with that. What specific service do you need? (e.g., oil change, tire rotation, jump start, flat tire)",
                        "What type of service are you looking for?",
                        "Please specify the service required."
                    ],
                     "appointment_location": [
                        "Where would you like to have the service done? You can provide an address for mobile service or ask for nearby shops.",
                        "Do you need mobile service at your location, or would you like to find a nearby shop?",
                        "Please provide the location for your service appointment."
                    ],
                     "appointment_date": [
                         "Okay, I have the service and vehicle details. What day works best for your appointment?",
                         "When would you like to schedule the appointment for?",
                         "Please suggest a date for your service."
                     ],
                     "appointment_time": [
                         "Got the date. Do you prefer morning, afternoon, or a specific time?",
                         "What time on {appointment_date} works for you?",
                         "Please provide your preferred time slot."
                     ],
                    "DEFAULT": [ # Fallback if specific slot template missing
                        "Could you please provide the {slot_name}?",
                        "What is the {slot_name}?",
                        "I need the {slot_name} to continue."
                    ]
                },
                "REQUEST_CONFIRMATION": [
                    "Okay, let's confirm: You need {intent_description} for a {vehicle_year} {vehicle_make} {vehicle_model} located at {pickup_location}{destination_info}{appointment_info}. Is this correct?",
                    "To summarize: Requesting {intent_description}. Vehicle: {vehicle_year} {vehicle_make} {vehicle_model}. Location: {pickup_location}{destination_info}{appointment_info}. Correct?",
                    "Please confirm the details: Service: {intent_description}. Car: {vehicle_year} {vehicle_make} {vehicle_model}. Where: {pickup_location}{destination_info}{appointment_info}. Is this all correct?"
                ],
                "RESPOND_COMPLETE": {
                     "towing": [
                         "Alright, your tow truck for the {vehicle_make} {vehicle_model} is booked! Help is on the way to {pickup_location} and should arrive in approximately 30-45 minutes.",
                         "Tow request confirmed! Your tow truck is dispatched to {pickup_location} for the {vehicle_make}.",
                         "Confirmed! A tow truck is en route to {pickup_location}. Expected arrival is within 45 minutes."
                     ],
                     "roadside": [
                         "Okay, help is on the way to {pickup_location} for your {service_type} request.",
                         "Confirmed! A technician has been dispatched to {pickup_location}.",
                         "Roadside assistance confirmed for {service_type} at {pickup_location}. Help will arrive soon."
                     ],
                     "appointment": [
                         "Your appointment for {service_type} on {appointment_date} at {appointment_time} at {appointment_location} is confirmed!",
                         "Great! You're all set for your {service_type} appointment on {appointment_date} at {appointment_time}.",
                         "Appointment confirmed! We look forward to seeing you for your {service_type} service."
                     ],
                     "DEFAULT": [
                         "Okay, your request is confirmed.",
                         "All set! Your request has been processed.",
                         "Confirmed!"
                     ]
                },
                "RESPOND_FALLBACK": {
                    "low_confidence": [
                        "Sorry, I didn't quite understand that. Could you please rephrase?",
                        "I'm not sure I got that. Can you say it differently?",
                        "Hmm, I need a bit more clarity. Could you provide more details?"
                    ],
                    "out_of_scope": [
                        "I can only help with towing, roadside assistance, or service appointments. How can I assist with one of those?",
                        "My apologies, that request is outside of my capabilities. I handle vehicle services like towing, roadside help, and appointments.",
                        "I understand you're asking about something else, but my expertise is in vehicle services. Can I help with towing, roadside, or an appointment?"
                    ],
                    "ambiguous": [
                        "I can help with several things. Are you looking for towing, roadside assistance, or to book a service appointment?",
                        "To help you best, could you specify if you need a tow, roadside help, or want to schedule service?",
                        "Please clarify the type of assistance you need: towing, roadside, or an appointment."
                    ],
                     "clarification_needed": [
                         "Could you please provide a bit more detail so I can assist you?",
                         "I need a little more information to understand your request.",
                         "To proceed, please tell me more about what you need."
                     ],
                    "nlu_error": [
                         "Sorry, I encountered an technical issue understanding your request. Could you please try rephrasing?",
                         "There was a problem processing your message. Please try again.",
                         "My apologies, I hit a snag. Can you repeat your request?"
                    ],
                     "internal_error": [
                          "Sorry, something went wrong on my end. Please try again.",
                          "I encountered an internal error. Let's try that again.",
                          "My apologies, I seem to be having technical difficulties."
                     ],
                    "DEFAULT": [
                         "I'm sorry, I didn't understand. Could you please rephrase?",
                         "Could you please provide more details?",
                         "I'm not sure how to help with that. Can you clarify?"
                    ]
                },
                 "RESPOND_RESTART_FLOW": [
                     "Okay, let's try that again. What information would you like to correct?",
                     "No problem. Let's re-enter the details. What would you like to change?",
                     "Understood. Let's restart the booking. What service do you need?" # Simple restart
                 ],
                 "RESPOND_ALREADY_COMPLETE": [
                     "Your request has already been confirmed.",
                     "We've already completed that request for you.",
                     "That booking is already finalized."
                 ]
            }

        def generate_response(self, action, state):
            """Generate a response based on the action and state."""
            action_type = action.get("type")
            response = "Sorry, I'm not sure how to respond to that." # Default

            if action_type in self.templates:
                templates_for_action = self.templates[action_type]

                if action_type == "REQUEST_SLOT":
                    slot_name = action.get("slot_name")
                    # Use specific template if available, else use DEFAULT
                    slot_templates = templates_for_action.get(slot_name, templates_for_action["DEFAULT"])
                    response = random.choice(slot_templates)
                    # Fill placeholders
                    response = response.replace("{slot_name}", slot_name.replace("_", " "))
                    # Fill context placeholders like {vehicle_make}
                    for key, value in state.entities.items():
                        response = response.replace(f"{{{key}}}", str(value))

                elif action_type == "REQUEST_CONFIRMATION":
                    details = action.get("details", {})
                    intent_desc = state.current_intent.replace("_", " ").replace("request ", "").replace("book ", "") # Basic description

                    # Build description strings, handling missing entities gracefully
                    vehicle_info = f"{details.get('vehicle_year','')} {details.get('vehicle_make','')} {details.get('vehicle_model','')}".strip()
                    if not vehicle_info: vehicle_info = "your vehicle"

                    destination_str = f" to {details.get('destination')}" if 'destination' in details else ""
                    pickup_str = f"{details.get('pickup_location', 'your location')}"

                    appointment_str = ""
                    if state.current_intent.startswith("appointment_"):
                        pickup_str = f"{details.get('appointment_location', 'your preferred location')}" # Appointment location
                        appointment_str = f" on {details.get('appointment_date', '[Date TBD]')} at {details.get('appointment_time', '[Time TBD]')}"

                    response = random.choice(templates_for_action)
                    response = response.format(
                         intent_description=intent_desc,
                         vehicle_year=details.get('vehicle_year',''),
                         vehicle_make=details.get('vehicle_make',''),
                         vehicle_model=details.get('vehicle_model',''),
                         pickup_location=pickup_str,
                         destination_info=destination_str,
                         appointment_info=appointment_str
                    ).replace("  "," ").strip() # Clean up extra spaces

                elif action_type == "RESPOND_COMPLETE":
                     details = action.get("details", {})
                     intent = action.get("intent", "")
                     sub_key = "DEFAULT" # Default subkey
                     if intent.startswith("towing"): sub_key = "towing"
                     elif intent.startswith("roadside"): sub_key = "roadside"
                     elif intent.startswith("appointment"): sub_key = "appointment"

                     response = random.choice(templates_for_action.get(sub_key, templates_for_action["DEFAULT"]))
                     # Fill placeholders from details
                     for key, value in details.items():
                         response = response.replace(f"{{{key}}}", str(value))
                     # Fill remaining known state entities if placeholders exist
                     for key, value in state.entities.items():
                         response = response.replace(f"{{{key}}}", str(value))


                elif action_type == "RESPOND_FALLBACK":
                    reason = action.get("reason", "DEFAULT")
                    # Use specific reason template if available, else use DEFAULT
                    fallback_templates = templates_for_action.get(reason, templates_for_action["DEFAULT"])
                    response = random.choice(fallback_templates)

                else: # For simple actions like RESPOND_RESTART_FLOW, RESPOND_ALREADY_COMPLETE
                     response = random.choice(templates_for_action)

            return response

    # --- End of ResponseGenerator Class ---
    ```

3.  **Integrate into `DialogManager`:**
    - Open `dialog_manager.py`.
    - Add the import: `from response_generator import ResponseGenerator`.
    - In the `DialogManager.__init__` method, add `self.response_generator = ResponseGenerator()`.
    - In the `DialogManager.process_turn` method, find the line `bot_response = f"Action: {action['type']}" # Placeholder response`. **Replace** this line with:
      ```python
      bot_response = self.response_generator.generate_response(action, state)
      ```
    - Save the changes to `dialog_manager.py`.

**Phase 3 Test:**

- **Instruction:** Create a new file named `test_phase_dialog_3.py`. Paste the following code into it. **DO NOT MODIFY THIS TEST SCRIPT.** Run it using `python test_phase_dialog_3.py`.
- **Debugging:** If the test fails:
  - Check for syntax errors or import errors in `response_generator.py` and `dialog_manager.py`.
  - Verify the `ResponseGenerator` class and `generate_response` method are defined correctly.
  - Ensure the response templates in `ResponseGenerator` cover all the `action["type"]` values produced by `DialogManager`. Check for typos in action types or template keys.
  - Add print statements inside `generate_response` to see which template is being chosen and how placeholders are being filled. Check if `state.entities` contains the expected values when filling placeholders.
  - Fix `response_generator.py` or the integration point in `dialog_manager.py`. Repeat until `test_phase_dialog_3.py` passes. Report success or failure.

```python
# test_phase_dialog_3.py
import unittest
import os
import sys

# Ensure root directory is in path to find inference and dialog_manager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock NLUInferencer for testing DialogManager structure without full NLU
class MockNLUInferencer:
    # Define predict to return different results based on input for testing various paths
    def predict(self, text):
        text_lower = text.lower()
        if "tow" in text_lower and "123 main" in text_lower and "abc auto" in text_lower and "honda civic 2018" in text_lower:
            return {"text": text, "intent": {"name": "towing_request_tow_full", "confidence": 0.99},
                    "entities": [{"entity":"pickup_location", "value":"123 Main St"}, {"entity":"destination", "value":"ABC Auto"},
                                 {"entity":"vehicle_make", "value":"Honda"}, {"entity":"vehicle_model", "value":"Civic"},
                                 {"entity":"vehicle_year", "value":"2018"}]}
        elif "tow" in text_lower:
            return {"text": text, "intent": {"name": "towing_request_tow_basic", "confidence": 0.9}, "entities": []}
        elif "battery" in text_lower:
             return {"text": text, "intent": {"name": "roadside_request_battery", "confidence": 0.85}, "entities": []}
        elif "appointment" in text_lower:
             return {"text": text, "intent": {"name": "appointment_book_service_basic", "confidence": 0.92}, "entities": []}
        elif "yes" in text_lower or "confirm" in text_lower: # Simulate confirmation
             return {"text": text, "intent": {"name": "affirm", "confidence": 0.99}, "entities": []}
        else:
            return {"text": text, "intent": {"name": "fallback_low_confidence", "confidence": 0.3}, "entities": []}


# Temporarily replace NLUInferencer during import
import inference
original_nlu_inferencer = inference.NLUInferencer
inference.NLUInferencer = MockNLUInferencer

# Now import DialogManager after mocking NLU
from dialog_manager import DialogState, DialogManager
from response_generator import ResponseGenerator # Import ResponseGenerator

# Restore original NLUInferencer if needed elsewhere
inference.NLUInferencer = original_nlu_inferencer


class TestPhase3ResponseGeneration(unittest.TestCase):

    def setUp(self):
        """Set up a new DialogManager for each test."""
        # We need to mock NLU for DialogManager tests
        self.patcher = patch('dialog_manager.NLUInferencer', MockNLUInferencer)
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
        self.assertIn("vehicle located", result["bot_response"].lower()) # Check for expected question
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
            "vehicle_year": "2020"
        }
        state.required_slots = ["pickup_location", "destination", "vehicle_make", "vehicle_model", "vehicle_year"]
        state.filled_slots = set(state.required_slots)
        state.current_step = "ASK_DUMMY" # Simulate being past the initial slot filling

        # Now process a turn that should lead to confirmation
        result = self.manager.process_turn("that's all the info", conv_id) # Input text content doesn't matter much here due to state override

        self.assertEqual(result["action"]["type"], "REQUEST_CONFIRMATION")
        self.assertIn("confirm", result["bot_response"].lower())
        self.assertIn("toyota camry", result["bot_response"].lower())
        self.assertIn("456 oak ave", result["bot_response"].lower())
        self.assertIn("city repair", result["bot_response"].lower())
        print(f"Response: {result['bot_response']}")
        print("Response Generation (Confirmation) Test PASSED.")

    def test_response_for_completion(self):
        """Test response generation upon task completion."""
        print("\nTesting Response Generation (Completion)...")
        conv_id = "resp_complete_1"
        # Simulate being at the confirmation step
        state = self.manager.get_or_create_state(conv_id)
        state.current_intent = "towing_request_tow_full"
        state.entities = {"pickup_location": "Work", "destination": "Home", "vehicle_make": "Kia", "vehicle_model": "Soul", "vehicle_year": "2021"}
        state.required_slots = list(state.entities.keys())
        state.filled_slots = set(state.required_slots)
        state.current_step = "CONFIRMATION"

        # User confirms
        result = self.manager.process_turn("yes confirm", conv_id)
        self.assertEqual(result["action"]["type"], "RESPOND_COMPLETE")
        self.assertIn("confirmed", result["bot_response"].lower())
        self.assertIn("booked", result["bot_response"].lower()) # Specific template for towing
        self.assertIn("kia soul", result["bot_response"].lower())
        print(f"Response: {result['bot_response']}")
        print("Response Generation (Completion) Test PASSED.")

    def test_response_for_fallback(self):
        """Test response generation for fallback scenarios."""
        print("\nTesting Response Generation (Fallback)...")
        conv_id = "resp_fallback_1"
        # Simulate low confidence NLU result
        result = self.manager.process_turn("garble blah", conv_id)
        self.assertEqual(result["action"]["type"], "RESPOND_FALLBACK")
        self.assertIn("sorry", result["bot_response"].lower())
        self.assertIn("understand", result["bot_response"].lower())
        print(f"Response: {result['bot_response']}")
        print("Response Generation (Fallback) Test PASSED.")


if __name__ == '__main__':
    print("--- Running Phase 3 Dialog Manager Tests ---")
    # Restore NLUInferencer before running tests if modified globally
    if 'original_nlu_inferencer' in globals():
        inference.NLUInferencer = original_nlu_inferencer
    unittest.main()
    # Restore again after tests if needed
    if 'original_nlu_inferencer' in globals():
        inference.NLUInferencer = original_nlu_inferencer
```

---

**Phase 4: Final Verification and Cleanup Check**

**Goal:** Ensure the core NLU + Dialog pipeline works end-to-end with the real models and that the project cleanup is complete.

**Steps:**

1.  **Run Integration Test:** Execute the _unmocked_ root `test_integration.py` again to ensure the previous phases haven't broken the core NLU predictions needed by the Dialog Manager.

    ```bash
    python test_integration.py
    ```

    - **Verify:** It MUST report "All tests PASSED!". If not, debug `inference.py` or `train.py` as per the previous Phase 3 instructions.

2.  **Run Cleanup Test:** Execute the cleanup verification script:
    ```bash
    python test_phase5.py
    ```
    - **Verify:** It MUST pass. If not, run `python cleanup.py` (or manually delete files) and update `.github/workflows/ci.yml` as needed, then rerun `test_phase5.py`.

**Phase 4 Test:**

- **Instruction:** The successful execution of `test_integration.py` (unmocked) and `test_phase5.py` constitutes passing this phase.
- **Debugging:** Address failures based on the specific test that fails, referring back to the debugging steps in previous phases.

---

**Final Confirmation:**

- After completing this revised 4-phase plan and passing all tests (including the unmocked `test_integration.py`), the system is ready. Confirm with: "All revised phases completed successfully. Simplified NLU system is functional, dialog management core logic is implemented, response generation is integrated, and project is cleaned. Ready for further development or testing."
