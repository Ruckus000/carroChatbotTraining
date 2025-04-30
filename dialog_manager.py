# dialog_manager.py
import copy

from inference import NLUInferencer  # Import the NLU class
from response_generator import ResponseGenerator


class DialogState:
    def __init__(self, conversation_id=None):
        self.conversation_id = conversation_id or "default_session"
        self.turn_count = 0
        self.history = []  # List of {"user": text, "bot": text}
        self.current_intent = None
        self.intent_confidence = 0.0
        self.entities = (
            {}
        )  # {entity_type: value} e.g. {"pickup_location": "123 Main St"}
        self.required_slots = []  # List of entity types needed for current intent
        self.filled_slots = set()  # Set of entity types already filled
        self.current_step = "START"  # Tracks progress within a flow
        self.fallback_reason = None  # 'low_confidence', 'out_of_scope', 'ambiguous'
        self.booking_details = {}  # Stores final booking info
        self.booking_confirmed = False

    def update_from_nlu(self, nlu_result):
        """Update state based on NLU output."""
        # Get the intent info
        intent_info = nlu_result.get("intent", {})
        new_intent = intent_info.get("name", "unknown")
        new_confidence = intent_info.get("confidence", 0.0)

        # Only update intent if we don't have one or if new intent has higher confidence
        if not self.current_intent or new_confidence > self.intent_confidence:
            self.current_intent = new_intent
            self.intent_confidence = new_confidence

        # Update entities, preferring new values
        nlu_entities = nlu_result.get("entities", [])
        for entity_info in nlu_entities:
            entity_type = entity_info.get("entity")
            entity_value = entity_info.get("value")
            if entity_type and entity_value:
                self.entities[entity_type] = entity_value
                self.filled_slots.add(entity_type)  # Mark as filled

        # Only set fallback if we're not in a flow
        if not self.required_slots:
            if new_intent == "fallback_low_confidence":
                self.fallback_reason = "low_confidence"
            elif new_intent.startswith("fallback_"):
                self.fallback_reason = "out_of_scope"
            elif new_intent.startswith("clarification_"):
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
        print(
            f"DEBUG: Resetting flow. Current state: {self.current_intent}, {self.current_step}"
        )
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
    def __init__(self, nlu_inferencer: NLUInferencer):
        """
        Initializes the DialogManager.

        Args:
            nlu_inferencer: An initialized instance of NLUInferencer.

        Raises:
            ValueError: If nlu_inferencer is not provided.
        """
        if not nlu_inferencer:
            raise ValueError("NLUInferencer instance is required for DialogManager.")
        self.nlu = nlu_inferencer
        self.states = (
            {}
        )  # Stores states per conversation_id: {conversation_id: DialogState}
        self.response_generator = ResponseGenerator()
        print("DEBUG: DialogManager initialized with provided NLUInferencer.")
        # DO NOT add any other initialization logic here.

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
            return [
                "pickup_location",
                "destination",
                "vehicle_make",
                "vehicle_model",
                "vehicle_year",
            ]
        elif intent == "roadside_request_battery":
            return ["pickup_location"]
        elif intent == "roadside_request_tire":
            return [
                "pickup_location",
                "vehicle_make",
                "vehicle_model",
            ]  # Need vehicle for tire equip
        elif intent in ["roadside_request_fuel", "roadside_request_keys"]:
            return ["pickup_location"]
        elif intent.startswith("roadside_"):  # General roadside
            return ["service_type", "pickup_location"]  # Ask for specific service first
        elif intent.startswith("appointment_"):
            return [
                "service_type",
                "appointment_location",
                "vehicle_make",
                "vehicle_model",
                "vehicle_year",
                "appointment_date",
                "appointment_time",
            ]
        else:
            return (
                []
            )  # No slots needed for fallback/clarification/basic intents initially

    def process_turn(self, user_input, conversation_id=None):
        """Process one turn of the conversation."""
        if not self.nlu:
            return {"response": "Error: NLU system not available.", "state": None}

        state = self.get_or_create_state(conversation_id)

        # Store previous state before updates
        previous_intent = state.current_intent
        previous_step = state.current_step

        # Increment turn count at the beginning
        state.turn_count += 1

        try:
            # Get NLU result - normal processing
            nlu_result = self.nlu.predict(user_input)
            print(f"DEBUG NLU Result: {nlu_result}")
        except Exception as e:
            print(f"ERROR during NLU prediction: {e}")
            state.fallback_reason = "nlu_error"
            nlu_result = {
                "intent": {"name": "fallback_nlu_error", "confidence": 1.0},
                "entities": [],
            }

        # Detect out-of-scope queries that are clearly not automotive related
        out_of_scope_patterns = [
            "chocolate",
            "cake",
            "bake",
            "weather",
            "recipe",
            "movie",
            "music",
            "politics",
            "sports",
            "game",
            "restaurant",
            "food",
        ]
        is_out_of_scope = any(
            pattern in user_input.lower() for pattern in out_of_scope_patterns
        )

        if is_out_of_scope:
            state.fallback_reason = "out_of_scope"
            state.current_intent = "fallback_out_of_scope"
        # Check for towing keywords only if not out of scope
        elif any(
            word in user_input.lower() for word in ["tow", "broke down", "broken down"]
        ):
            state.current_intent = "towing_request_tow_basic"
            state.required_slots = self.define_required_slots(
                "towing_request_tow_basic"
            )
            state.fallback_reason = None  # Clear any fallback since we detected towing

        # Update State with NLU result
        state.update_from_nlu(nlu_result)

        # If we're in a towing flow, maintain context even with low confidence
        if previous_intent and previous_intent.startswith("towing_"):
            if state.fallback_reason == "low_confidence":
                # Clear fallback and maintain towing context
                state.fallback_reason = None
                state.current_intent = previous_intent

        # If no intent is set yet but we got entities, use a generic intent
        if not state.current_intent or state.current_intent == "unknown":
            if nlu_result.get("entities"):
                state.current_intent = "entity_only"

        # Set required slots if needed
        if state.current_intent and not state.required_slots:
            state.required_slots = self.define_required_slots(state.current_intent)

        # Determine next action based on updated state
        action = self.determine_next_action(state, state.current_intent, user_input)
        print(f"DEBUG Action decided: {action}")

        # Generate response based on action
        bot_response = self.response_generator.generate_response(action, state)

        # Add the interaction to history
        state.add_history(user_input, bot_response)

        # Return the complete result
        return {"state": state, "action": action, "bot_response": bot_response}

    def determine_next_action(self, state, nlu_intent, user_input):
        """Determine the next action based on state and NLU intent."""
        if state.booking_confirmed:
            return {"type": "RESPOND_ALREADY_COMPLETE"}

        # Handle restart flow (user wants to start over)
        if nlu_intent.startswith("restart") or nlu_intent.startswith("cancel"):
            state.reset_flow()  # Reset the state
            return {"type": "RESPOND_RESTART_FLOW"}

        # Handle clearly out-of-scope queries
        out_of_scope_patterns = [
            "chocolate",
            "cake",
            "bake",
            "weather",
            "recipe",
            "movie",
            "music",
            "politics",
            "sports",
            "game",
            "restaurant",
            "food",
        ]

        if any(pattern in user_input.lower() for pattern in out_of_scope_patterns):
            return {"type": "RESPOND_FALLBACK", "reason": "out_of_scope"}

        # Only fallback if we have no context and get a low confidence result
        if state.fallback_reason and not state.current_intent:
            return {"type": "RESPOND_FALLBACK", "reason": state.fallback_reason}

        # Initial case with no intent yet - try to detect one
        if not state.current_intent or state.current_intent == "unknown":
            # Check for towing keywords in the input
            if any(
                word in user_input.lower()
                for word in ["tow", "broke down", "broken down"]
            ):
                state.current_intent = "towing_request_tow_basic"
                state.required_slots = self.define_required_slots(state.current_intent)
            else:
                # Assign the intent detected by NLU
                state.current_intent = nlu_intent

        # -----------------------------------------------
        # VEHICLE TOWING FLOW
        # -----------------------------------------------
        if state.current_intent.startswith("towing_"):
            # Set required slots for towing if not already set
            if not state.required_slots:
                state.required_slots = self.define_required_slots(state.current_intent)
                state.current_step = "COLLECTING_INFO"

            # Check if we have all required slots
            missing_slots = state.get_missing_slots()
            if missing_slots:
                # Ask for the first missing slot
                slot_to_request = missing_slots[0]
                state.current_step = f"ASK_{slot_to_request.upper()}"
                return {"type": "REQUEST_SLOT", "slot_name": slot_to_request}
            else:
                # All slots filled, move to confirmation
                if state.current_step != "CONFIRMATION":
                    state.current_step = "CONFIRMATION"
                    return {
                        "type": "REQUEST_CONFIRMATION",
                        "details": state.entities,
                        "intent": state.current_intent,
                    }
                else:
                    # Confirmation step completed, move to booking
                    state.current_step = "BOOKING"
                    state.booking_confirmed = True
                    return {
                        "type": "RESPOND_COMPLETE",
                        "details": state.entities,
                        "intent": "towing",
                    }

        # -----------------------------------------------
        # ROADSIDE ASSISTANCE FLOW
        # -----------------------------------------------
        elif state.current_intent.startswith("roadside_"):
            # Set required slots for roadside if not already set
            if not state.required_slots:
                state.required_slots = self.define_required_slots(state.current_intent)
                state.current_step = "COLLECTING_INFO"

            # Check if we have all required slots
            missing_slots = state.get_missing_slots()
            if missing_slots:
                # Ask for the first missing slot
                slot_to_request = missing_slots[0]
                state.current_step = f"ASK_{slot_to_request.upper()}"
                return {"type": "REQUEST_SLOT", "slot_name": slot_to_request}
            else:
                # All slots filled, move to confirmation
                if state.current_step != "CONFIRMATION":
                    state.current_step = "CONFIRMATION"
                    return {
                        "type": "REQUEST_CONFIRMATION",
                        "details": state.entities,
                        "intent": state.current_intent,
                    }
                else:
                    # Confirmation step completed, move to booking
                    state.current_step = "BOOKING"
                    state.booking_confirmed = True
                    intent_type = (
                        "roadside"
                        if state.current_intent.startswith("roadside")
                        else "unknown"
                    )
                    # Extract service type or default to generic roadside
                    service_type = state.entities.get(
                        "service_type", state.current_intent.replace("roadside_", "")
                    )
                    state.entities["service_type"] = service_type
                    return {
                        "type": "RESPOND_COMPLETE",
                        "details": state.entities,
                        "intent": intent_type,
                    }

        # -----------------------------------------------
        # SERVICE APPOINTMENT FLOW
        # -----------------------------------------------
        elif state.current_intent.startswith("appointment_"):
            # Set required slots for appointment if not already set
            if not state.required_slots:
                state.required_slots = self.define_required_slots(state.current_intent)
                state.current_step = "COLLECTING_INFO"

            # Check if we have all required slots
            missing_slots = state.get_missing_slots()
            if missing_slots:
                # Ask for the first missing slot
                slot_to_request = missing_slots[0]
                state.current_step = f"ASK_{slot_to_request.upper()}"
                return {"type": "REQUEST_SLOT", "slot_name": slot_to_request}
            else:
                # All slots filled, move to confirmation
                if state.current_step != "CONFIRMATION":
                    state.current_step = "CONFIRMATION"
                    return {
                        "type": "REQUEST_CONFIRMATION",
                        "details": state.entities,
                        "intent": state.current_intent,
                    }
                else:
                    # Confirmation step completed, move to booking
                    state.current_step = "BOOKING"
                    state.booking_confirmed = True
                    return {
                        "type": "RESPOND_COMPLETE",
                        "details": state.entities,
                        "intent": "appointment",
                    }

        # -----------------------------------------------
        # FALLBACK BEHAVIOR
        # -----------------------------------------------
        else:
            # If we have any required slots, we're in a flow - continue it
            if state.required_slots:
                missing_slots = state.get_missing_slots()
                if missing_slots:
                    return {"type": "REQUEST_SLOT", "slot_name": missing_slots[0]}

            # Otherwise, need clarification
            return {
                "type": "RESPOND_FALLBACK",
                "reason": "clarification_needed",
            }

    def generate_response_for_action(self, state, action):
        """Generate the response based on the action."""
        # Handle direct towing response for the hack added above
        if action.get("type") == "RESPOND_WITH_TOWING":
            return action.get("text")
        # Handle location confirmation
        elif action.get("type") == "RESPOND_WITH_TOWING_LOCATION":
            return action.get("text")

        # Generate responses for normal actions
        response = self.response_generator.generate_response(action, state)
        return response


# --- End of DialogManager Class ---
