# dialog_manager.py
import copy
from inference import NLUInferencer # Import the NLU class

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