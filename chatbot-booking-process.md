# Chatbot Booking Process Flow Outline

## Common Initial Steps (All Booking Flows)

1. **Greeting and Service Identification**
   - Welcome user
   - Identify user intent
   - Determine appropriate booking flow (Towing, Roadside Assistance, Service Appointment)

2. **Context Awareness Processing**
   - Run negation detection to identify if user is canceling/declining
   - Run context switch detection to identify if user is changing request type
   - Respond appropriately to negations or context switches

## Towing Service Flow

1. **Gather Vehicle Information**
   - Vehicle make, model, and year
   - Vehicle color
   - License plate (optional)

2. **Gather Location Information**
   - Current vehicle location
   - Desired destination for tow
   - Verification of location details

3. **Gather Situation Details**
   - Nature of the breakdown
   - Vehicle condition
   - Special towing requirements (e.g., flatbed needed)

4. **Confirm Towing Request**
   - Present summary of tow request details
   - Get user confirmation
   - Inform about ETA and next steps

5. **Towing Service Updates**
   - Provide tow truck ETA
   - Update on tow truck arrival
   - Confirm completion of service
   - Follow-up for feedback

## Roadside Assistance Flow

1. **Identify Specific Roadside Need**
   - Dead battery/Jump start
   - Flat tire
   - Locked out
   - Fuel delivery
   - Other mechanical issues

2. **Gather Vehicle Information**
   - Vehicle make, model, and year
   - Vehicle color
   - License plate (optional)

3. **Gather Location Information**
   - Current vehicle location
   - Additional landmark details
   - Verification of location

4. **Confirm Roadside Assistance Request**
   - Present summary of assistance request
   - Get user confirmation
   - Inform about ETA and next steps

5. **Roadside Service Updates**
   - Provide technician ETA
   - Update on technician arrival
   - Confirm completion of service
   - Follow-up for feedback
   - Upgrade to towing if roadside assistance insufficient

## Service Appointment Flow

1. **Identify Service Type**
   - Regular maintenance
   - Specific repair
   - Diagnostic service
   - Inspection

2. **Gather Vehicle Information**
   - Vehicle make, model, and year
   - Current mileage
   - Service history (if available)

3. **Schedule Service**
   - Propose available dates/times
   - Get user preference
   - Confirm appointment time

4. **Gather Additional Information**
   - Customer contact details
   - Special requests
   - Transportation needs during service

5. **Confirm Appointment**
   - Present appointment summary
   - Get user confirmation
   - Provide next steps and preparation info

6. **Appointment Reminders and Updates**
   - Send appointment reminder
   - Notify of any changes
   - Follow-up post-service

## Exception Handling (All Flows)

1. **Handle Negations**
   - Detect cancellation or negative responses
   - Confirm cancellation
   - Offer alternatives

2. **Handle Context Switches**
   - Detect when user changes service type
   - Seamlessly transition to new flow
   - Retain relevant information already gathered

3. **Handle Incomplete Information**
   - Recognize missing critical details
   - Re-prompt for required information
   - Offer examples or clarification

4. **Handle Special Requirements**
   - Emergency priority handling
   - Accessibility needs
   - After-hours service requirements

## Closing Steps (All Flows)

1. **Verification**
   - Confirm all details are accurate
   - Provide summary of the booking
   - Clarify next steps

2. **Information Sharing**
   - Share booking reference number
   - Provide contact details for follow-up
   - Explain how to track/modify booking

3. **User Feedback**
   - Ask for initial satisfaction
   - Set expectation for follow-up survey
   - Thank user for their business
