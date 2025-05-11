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
                    "To proceed, please provide the pickup location for your vehicle.",
                ],
                "destination": [
                    "Thanks! And where does the vehicle need to be towed to?",
                    "What's the destination address or shop name?",
                    "Please provide the drop-off location.",
                ],
                "vehicle_make": [
                    "Okay, I have the locations. What is the make of your vehicle?",
                    "Can you tell me the make of the car?",
                    "What vehicle make should I note down?",
                ],
                "vehicle_model": [
                    "Thanks. And the model?",
                    "What model is the {vehicle_make}?",  # Example using context
                    "Please provide the vehicle model.",
                ],
                "vehicle_year": [
                    "Almost done with vehicle details. What year is the {vehicle_make} {vehicle_model}?",
                    "What's the year of the vehicle?",
                    "Please tell me the vehicle year.",
                ],
                "service_type": [  # For general roadside or appointment start
                    "I can help with that. What specific service do you need? (e.g., oil change, tire rotation, jump start, flat tire)",
                    "What type of service are you looking for?",
                    "Please specify the service required.",
                ],
                "appointment_location": [
                    "Where would you like to have the service done? You can provide an address for mobile service or ask for nearby shops.",
                    "Do you need mobile service at your location, or would you like to find a nearby shop?",
                    "Please provide the location for your service appointment.",
                ],
                "appointment_date": [
                    "Okay, I have the service and vehicle details. What day works best for your appointment?",
                    "When would you like to schedule the appointment for?",
                    "Please suggest a date for your service.",
                ],
                "appointment_time": [
                    "Got the date. Do you prefer morning, afternoon, or a specific time?",
                    "What time on {appointment_date} works for you?",
                    "Please provide your preferred time slot.",
                ],
                "DEFAULT": [  # Fallback if specific slot template missing
                    "Could you please provide the {slot_name}?",
                    "What is the {slot_name}?",
                    "I need the {slot_name} to continue.",
                ],
            },
            "REQUEST_CONFIRMATION": [
                "Okay, let's confirm: You need {intent_description} for a {vehicle_year} {vehicle_make} {vehicle_model} located at {pickup_location}{destination_info}{appointment_info}. Is this correct?",
                "To summarize: Requesting {intent_description}. Vehicle: {vehicle_year} {vehicle_make} {vehicle_model}. Location: {pickup_location}{destination_info}{appointment_info}. Correct?",
                "Please confirm the details: Service: {intent_description}. Car: {vehicle_year} {vehicle_make} {vehicle_model}. Where: {pickup_location}{destination_info}{appointment_info}. Is this all correct?",
            ],
            "RESPOND_COMPLETE": {
                "towing": [
                    "Alright, your tow truck for the {vehicle_make} {vehicle_model} is booked! Help is on the way to {pickup_location} and should arrive in approximately 30-45 minutes.",
                    "Tow request confirmed! Your tow truck is dispatched to {pickup_location} for the {vehicle_make}.",
                    "Confirmed! A tow truck is en route to {pickup_location}. Expected arrival is within 45 minutes.",
                ],
                "roadside": [
                    "Okay, help is on the way to {pickup_location} for your {service_type} request.",
                    "Confirmed! A technician has been dispatched to {pickup_location}.",
                    "Roadside assistance confirmed for {service_type} at {pickup_location}. Help will arrive soon.",
                ],
                "appointment": [
                    "Your appointment for {service_type} on {appointment_date} at {appointment_time} at {appointment_location} is confirmed!",
                    "Great! You're all set for your {service_type} appointment on {appointment_date} at {appointment_time}.",
                    "Appointment confirmed! We look forward to seeing you for your {service_type} service.",
                ],
                "DEFAULT": [
                    "Okay, your request is confirmed.",
                    "All set! Your request has been processed.",
                    "Confirmed!",
                ],
            },
            "RESPOND_FALLBACK": {
                "low_confidence": [
                    "Hello! I'm your automotive assistant. I can help with towing, roadside assistance, or scheduling service appointments. What can I help you with today?",
                    "Hi there! I'm here to assist with vehicle services like towing, roadside help, or service appointments. What do you need help with?",
                    "Welcome! I can help with your automotive needs. Do you need towing, roadside assistance, or would you like to schedule a service appointment?",
                ],
                "out_of_scope": [
                    "I can only help with towing, roadside assistance, or service appointments. How can I assist with one of those?",
                    "My apologies, that request is outside of my capabilities. I handle vehicle services like towing, roadside help, and appointments.",
                    "I understand you're asking about something else, but my expertise is in vehicle services. Can I help with towing, roadside, or an appointment?",
                ],
                "ambiguous": [
                    "I can help with several things. Are you looking for towing, roadside assistance, or to book a service appointment?",
                    "To help you best, could you specify if you need a tow, roadside help, or want to schedule service?",
                    "Please clarify the type of assistance you need: towing, roadside, or an appointment.",
                ],
                "clarification_needed": [
                    "Could you please provide a bit more detail so I can assist you?",
                    "I need a little more information to understand your request.",
                    "To proceed, please tell me more about what you need.",
                ],
                "nlu_error": [
                    "Sorry, I encountered an technical issue understanding your request. Could you please try rephrasing?",
                    "There was a problem processing your message. Please try again.",
                    "My apologies, I hit a snag. Can you repeat your request?",
                ],
                "internal_error": [
                    "Sorry, something went wrong on my end. Please try again.",
                    "I encountered an internal error. Let's try that again.",
                    "My apologies, I seem to be having technical difficulties.",
                ],
                "DEFAULT": [
                    "I'm sorry, I didn't understand. Could you please rephrase?",
                    "Could you please provide more details?",
                    "I'm not sure how to help with that. Can you clarify?",
                ],
            },
            "RESPOND_RESTART_FLOW": [
                "Okay, let's try that again. What information would you like to correct?",
                "No problem. Let's re-enter the details. What would you like to change?",
                "Understood. Let's restart the booking. What service do you need?",  # Simple restart
            ],
            "RESPOND_ALREADY_COMPLETE": [
                "Your request has already been confirmed.",
                "We've already completed that request for you.",
                "That booking is already finalized.",
            ],
        }

    def generate_response(self, action, state):
        """Generate a response based on the action and state."""
        action_type = action.get("type")
        response = "Sorry, I'm not sure how to respond to that."  # Default

        # Only use special handling for the first message
        if state.turn_count <= 1 and state.history and "user" in state.history[-1]:
            user_text = state.history[-1]["user"].lower()
            if (
                "tow" in user_text
                or "broke down" in user_text
                or "car won't start" in user_text
                or "flat tire" in user_text
            ):
                # Force a towing response for first message only
                return "I can help you with towing your vehicle. To get started, please provide your current location."

            if (
                "battery" in user_text
                or "jump start" in user_text
                or "won't turn on" in user_text
            ):
                return "I can send someone to help with your battery issue. Can you provide your current location?"

        if action_type in self.templates:
            templates_for_action = self.templates[action_type]

            if action_type == "REQUEST_SLOT":
                slot_name = action.get("slot_name")
                # Use specific template if available, else use DEFAULT
                slot_templates = templates_for_action.get(
                    slot_name, templates_for_action["DEFAULT"]
                )
                response = random.choice(slot_templates)
                # Fill placeholders
                response = response.replace("{slot_name}", slot_name.replace("_", " "))
                # Fill context placeholders like {vehicle_make}
                for key, value in state.entities.items():
                    response = response.replace(f"{{{key}}}", str(value))

            elif action_type == "REQUEST_CONFIRMATION":
                details = action.get("details", {})
                intent_desc = (
                    state.current_intent.replace("_", " ")
                    .replace("request ", "")
                    .replace("book ", "")
                )  # Basic description

                # Build description strings, handling missing entities gracefully
                vehicle_info = f"{details.get('vehicle_year','')} {details.get('vehicle_make','')} {details.get('vehicle_model','')}".strip()
                if not vehicle_info:
                    vehicle_info = "your vehicle"

                destination_str = (
                    f" to {details.get('destination')}"
                    if "destination" in details
                    else ""
                )
                pickup_str = f"{details.get('pickup_location', 'your location')}"

                appointment_str = ""
                if state.current_intent.startswith("appointment_"):
                    pickup_str = f"{details.get('appointment_location', 'your preferred location')}"  # Appointment location
                    appointment_str = f" on {details.get('appointment_date', '[Date TBD]')} at {details.get('appointment_time', '[Time TBD]')}"

                response = random.choice(templates_for_action)
                response = (
                    response.format(
                        intent_description=intent_desc,
                        vehicle_year=details.get("vehicle_year", ""),
                        vehicle_make=details.get("vehicle_make", ""),
                        vehicle_model=details.get("vehicle_model", ""),
                        pickup_location=pickup_str,
                        destination_info=destination_str,
                        appointment_info=appointment_str,
                    )
                    .replace("  ", " ")
                    .strip()
                )  # Clean up extra spaces

            elif action_type == "RESPOND_COMPLETE":
                details = action.get("details", {})
                intent = action.get("intent", "")
                sub_key = "DEFAULT"  # Default subkey
                if intent.startswith("towing"):
                    sub_key = "towing"
                elif intent.startswith("roadside"):
                    sub_key = "roadside"
                elif intent.startswith("appointment"):
                    sub_key = "appointment"

                response = random.choice(
                    templates_for_action.get(sub_key, templates_for_action["DEFAULT"])
                )
                # Fill placeholders from details
                for key, value in details.items():
                    response = response.replace(f"{{{key}}}", str(value))
                # Fill remaining known state entities if placeholders exist
                for key, value in state.entities.items():
                    response = response.replace(f"{{{key}}}", str(value))

            elif action_type == "RESPOND_FALLBACK":
                reason = action.get("reason", "DEFAULT")
                # Use specific reason template if available, else use DEFAULT
                fallback_templates = templates_for_action.get(
                    reason, templates_for_action["DEFAULT"]
                )
                response = random.choice(fallback_templates)

            else:  # For simple actions like RESPOND_RESTART_FLOW, RESPOND_ALREADY_COMPLETE
                response = random.choice(templates_for_action)

        return response


# --- End of ResponseGenerator Class ---
