Okay, let's create a highly structured and constrained implementation plan designed to minimize Cursor's deviation while allowing for necessary, controlled adjustments. This plan emphasizes explicitness, verification, and strict adherence to scope for each step.

**Execution Rules for Cursor:**

1.  **Strict Sequential Execution:** Execute phases and steps **strictly** in the order presented. Do **NOT** skip steps or jump between phases.
2.  **Targeted Modifications ONLY:** Modify **ONLY** the files and specific functions/lines mentioned in each step. Do **NOT** modify any other files or code sections.
3.  **No Unsolicited Code:** Do **NOT** add any new features, classes, functions, complex logic, or optimizations beyond precisely what is described in the step. Adhere strictly to the provided code snippets or implementation descriptions.
4.  **Mandatory Verification:** After completing the action(s) in a step, perform **ALL** verification actions for that step. Do **NOT** proceed if verification fails.
5.  **Test Integrity:** Execute test commands exactly as written. **DO NOT MODIFY ANY TEST SCRIPTS** (`test_*.py`) unless explicitly instructed by a _future_ plan revision _after_ reporting a fundamental issue with a test.
6.  **Limited Debugging Protocol:**
    - If a **command fails** (e.g., `python`, `pytest`), report the full command and error message and STOP.
    - If a **verification step fails**, report the failed verification and the actual outcome, then STOP.
    - If a **test fails**, report the full test output. Review **ONLY** the code modified in the _immediately preceding action steps_ of the _current phase_. Attempt **one** targeted fix based _only_ on the plan's instructions for that code. Rerun the test. If it still fails, STOP and report the failure and the attempted fix. Do **NOT** attempt broad debugging or modify unrelated code.
7.  **STOP on Error/Uncertainty:** If any command fails, verification fails, a test fails after one targeted fix attempt, or if instructions are unclear/ambiguous, **STOP EXECUTION IMMEDIATELY.**
8.  **Clear Reporting:** When Stopping (Rule 7), report:
    - The Phase and Step number where the issue occurred.
    - The exact command run or verification attempted.
    - The FULL error message, verification failure details, or test failure output.
    - Confirm that **ONLY** the files/code specified in the preceding steps of the current phase were modified.
9.  **Confirm Phase Completion:** After successfully completing all steps in a phase, including passing all associated tests and verifications, explicitly state: "Phase [X] completed successfully. All actions performed and verified."

---

**The Implementation Plan**

**Phase 0: Preparation**

- **Action 0.1:** Execute `git status` to ensure there are no uncommitted changes. If there are, STOP and report.
- **Action 0.2:** Confirm the following files exist: `dialog_manager.py`, `api.py`, `inference.py`, `response_generator.py`, `test_phase_dialog_2.py` (or similar existing dialog test).
- **Verification 0.2:** Report confirmation that files exist.

**Phase 1: Refactor `DialogManager`**

- **Goal:** Modify `DialogManager` for NLU dependency injection and ensure its core logic can handle unified flows (verified via tests later).

- **Step 1.1: Modify `DialogManager.__init__` for NLU Injection**

  - **Action:** Open `dialog_manager.py`. Modify the `__init__` method signature and body as follows:

    ```python
    # Add NLUInferencer to imports if not already there
    from inference import NLUInferencer
    from response_generator import ResponseGenerator # Ensure this is imported

    class DialogManager:
        # Replace the existing __init__ method with this:
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
            self.states = {}  # Stores states per conversation_id: {conversation_id: DialogState}
            self.response_generator = ResponseGenerator()
            print("DEBUG: DialogManager initialized with provided NLUInferencer.")
            # DO NOT add any other initialization logic here.
    ```

  - **Verification:** Manually review the diff for `dialog_manager.py`. Confirm that _only_ the `__init__` method was changed as shown, and the old NLU instantiation logic inside `__init__` is removed. Confirm `ResponseGenerator` is imported and instantiated.

- **Step 1.2: Verify `define_required_slots`**

  - **Action:** Open `dialog_manager.py`. Locate the `define_required_slots` method.
  - **Verification:**
    - Check if the `if intent.startswith("towing_"):` block returns a list containing at least: `"pickup_location"`, `"destination"`, `"vehicle_make"`, `"vehicle_model"`, `"vehicle_year"`.
    - Check if other intent prefixes (`roadside_`, `appointment_`) have corresponding reasonable slot lists.
    - Report confirmation or any discrepancies found (but DO NOT fix discrepancies yet).

- **Step 1.3: Create `DialogManager` Unit Tests**

  - **Action:** Create a new file named `test_dialog_manager_unified.py`. Add the following test structure (you will fill in assertions later):

    ```python
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
                DialogManager(nlu_inferencer=None) # Test None case
            print("PASSED")

        def test_towing_slot_definitions(self):
            """Verify required slots for towing intents."""
            print("\nRunning test: test_towing_slot_definitions")
            slots = self.manager.define_required_slots("towing_request_tow_full")
            expected_slots = ["pickup_location", "destination", "vehicle_make", "vehicle_model", "vehicle_year"]
            # Use assertCountEqual for order-independent list comparison
            self.assertCountEqual(slots, expected_slots, f"Expected {expected_slots}, got {slots}")
            print("PASSED")

        def test_full_towing_conversation_flow(self):
            """Simulate a complete happy-path towing conversation."""
            print("\nRunning test: test_full_towing_conversation_flow")
            conv_id = "tow_happy_path"
            user_inputs = [
                "I need a tow",
                "123 Main St",
                "ABC Auto Shop",
                "It's a 2019 Honda Civic",
                "Yes, that's right"
            ]
            # Define mock NLU responses corresponding to each user input
            mock_nlu_responses = [
                {"intent": {"name": "towing_request_tow_basic", "confidence": 0.95}, "entities": []},
                {"intent": {"name": "entity_only", "confidence": 0.9}, "entities": [{"entity": "pickup_location", "value": "123 Main St"}]},
                {"intent": {"name": "entity_only", "confidence": 0.9}, "entities": [{"entity": "destination", "value": "ABC Auto Shop"}]},
                {"intent": {"name": "entity_only", "confidence": 0.9}, "entities": [
                    {"entity": "vehicle_year", "value": "2019"},
                    {"entity": "vehicle_make", "value": "Honda"},
                    {"entity": "vehicle_model", "value": "Civic"}
                ]},
                {"intent": {"name": "affirm", "confidence": 0.98}, "entities": []}
            ]
            expected_action_types = [
                "REQUEST_SLOT", # Ask location
                "REQUEST_SLOT", # Ask destination
                "REQUEST_SLOT", # Ask make (or year/model depending on impl)
                "REQUEST_CONFIRMATION", # Ask confirmation
                "RESPOND_COMPLETE"  # Final response
            ]
            expected_next_slots = [ # What slot should it ask for next?
                "pickup_location",
                "destination",
                "vehicle_make", # Assuming this order
                None, # Confirmation step
                None  # Completion step
            ]

            self.assertEqual(len(user_inputs), len(mock_nlu_responses))
            self.assertEqual(len(user_inputs), len(expected_action_types))

            # Monkey patch the predict method for this test case
            self.mock_nlu.predict = MagicMock()
            self.mock_nlu.predict.side_effect = mock_nlu_responses

            state = None
            for i, user_input in enumerate(user_inputs):
                print(f"  Turn {i+1}: User says '{user_input}'")
                result = self.manager.process_turn(user_input, conv_id)
                state = result["state"] # Get updated state for next turn / assertions
                action = result["action"]
                bot_response = result["bot_response"] # Get response for logging

                print(f"  Bot Action: {action['type']}, Response: '{bot_response}'")

                # Verify the action type for this turn
                self.assertEqual(action["type"], expected_action_types[i], f"Turn {i+1}: Unexpected action type")

                # Verify the slot being requested (if applicable)
                if expected_action_types[i] == "REQUEST_SLOT":
                    self.assertEqual(action.get("slot_name"), expected_next_slots[i], f"Turn {i+1}: Unexpected slot requested")
                elif expected_action_types[i] == "REQUEST_CONFIRMATION":
                     # Check if all required slots are filled before confirmation
                     missing = state.get_missing_slots()
                     self.assertEqual(len(missing), 0, f"Turn {i+1}: Still missing slots before confirmation: {missing}")
                elif expected_action_types[i] == "RESPOND_COMPLETE":
                     self.assertTrue(state.booking_confirmed, f"Turn {i+1}: Booking not confirmed after completion action")

            # Final check: Ensure NLU predict was called correctly for each input
            self.assertEqual(self.mock_nlu.predict.call_count, len(user_inputs))
            print("PASSED")

    if __name__ == '__main__':
        unittest.main()
    ```

  - **Verification:** Confirm the file `test_dialog_manager_unified.py` exists and contains the code structure above.

- **Step 1.4: Run `DialogManager` Unit Tests**

  - **Action:** Execute `python -m unittest test_dialog_manager_unified.py`
  - **Verification:** Check the output. All tests MUST pass. If `test_full_towing_conversation_flow` fails, analyze the failure trace based _only_ on the logic within `DialogManager` (specifically `process_turn`, `determine_next_action`, `update_from_nlu`, `get_missing_slots`). Apply **one** targeted fix to `DialogManager` logic if needed and rerun the test. If it still fails, STOP (Rule 6).

- **Confirmation:** State "Phase 1 completed successfully. All actions performed and verified."

**Phase 2: Refactor `api.py`**

- **Goal:** Remove hardcoded logic and state from `api.py`, delegating fully to `DialogManager`.

- **Step 2.1: Remove `TowingState` Class**

  - **Action:** Open `api.py`. Delete the entire class definition `class TowingState: ...`.
  - **Verification:** Confirm the `TowingState` class definition is no longer present in `api.py`.

- **Step 2.2: Remove Hardcoded Towing Logic**

  - **Action:** Open `api.py`. Locate the `process_dialog` function. Delete the entire `if/elif/else` block that checks `cid in app.towing_state.locations` or `towing_keywords` and directly returns responses or modifies `app.towing_state`. This block likely starts _after_ `cid = request.conversation_id or "default"` and ends just before the `if dialog_manager:` check (or where the `dialog_manager.process_turn` call will be placed). _Be precise in deleting only this block._
  - **Verification:** Manually review `process_dialog`. Confirm the hardcoded towing logic (checking keywords, managing `app.towing_state`, returning direct confirmations/location requests) is gone.

- **Step 2.3: Implement NLU/DM Instantiation and Injection**

  - **Action:** Open `api.py`. Ensure the NLU and DialogManager are instantiated _once_ at the module level (outside request functions), and the NLU instance is passed to the DialogManager, like the example in Action 2.3 of the previous plan. Add necessary imports (`NLUInferencer`, `DialogManager`). Ensure logger uses `__name__`.

    ```python
    # api.py (near top, after imports and basic logging config)
    import logging
    from inference import NLUInferencer
    from dialog_manager import DialogManager
    # ... other imports like FastAPI, BaseModel, Optional, uuid etc.

    logger = logging.getLogger(__name__) # Use __name__

    # Initialize NLU model (once)
    try:
        nlu_inferencer_instance = NLUInferencer()
        logger.info("NLU model loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal: Failed to load NLU model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load NLU model: {e}") # Fail fast on startup

    # Initialize Dialog Manager (once) with NLU instance
    try:
        # Ensure the global dialog_manager variable is used by the endpoint
        dialog_manager = DialogManager(nlu_inferencer=nlu_inferencer_instance)
        logger.info("Dialog Manager initialized successfully.")
    except Exception as e:
        logger.error(f"Fatal: Failed to initialize Dialog Manager: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Dialog Manager: {e}") # Fail fast

    # (FastAPI app = FastAPI(...) setup, middleware, etc.)
    ```

  - **Verification:** Review the top level of `api.py`. Confirm `nlu_inferencer_instance` and `dialog_manager` are created once. Confirm `dialog_manager` receives `nlu_inferencer_instance`.

- **Step 2.4: Implement Simplified `/api/dialog` Endpoint**

  - **Action:** Open `api.py`. Replace the _entire body_ of the `process_dialog` async function with the following implementation. Ensure necessary imports (`DialogRequest`, `DialogResponse`, `uuid`, `BaseModel`, `Optional`) are present.

    ```python
    # Ensure these models are defined (likely already are)
    from pydantic import BaseModel
    from typing import Optional, Dict, Any # Add Dict, Any if process_turn returns complex dict
    import uuid

    class DialogRequest(BaseModel):
        text: str
        conversation_id: Optional[str] = None

    class DialogResponse(BaseModel):
        text: str
        conversation_id: str

    # Replace the existing process_dialog function
    @app.post("/api/dialog", response_model=DialogResponse)
    async def process_dialog(request: DialogRequest):
        # Use provided ID or generate a new one
        # Use str(uuid.uuid4()) for guaranteed string format if needed elsewhere
        conversation_id = request.conversation_id if request.conversation_id else str(uuid.uuid4())
        logger.info(
            f"Processing dialog for conv_id '{conversation_id}': '{request.text}'"
        )

        try:
            # === Main Change: Delegate directly to the single DialogManager instance ===
            # Assume process_turn returns a dict like: {"bot_response": str, "state": DialogState, "action": dict}
            # Adjust parsing based on actual DialogManager.process_turn return value
            dm_result: Dict[str, Any] = dialog_manager.process_turn(request.text, conversation_id)

            # Safely extract the response text
            if isinstance(dm_result, dict) and "bot_response" in dm_result:
                response_text = dm_result.get("bot_response", "Error: No response generated.")
                # Optional: Log state or action details for debugging
                # logger.debug(f"DM Action for {conversation_id}: {dm_result.get('action')}")
            else:
                # Handle cases where DialogManager might return None or an unexpected structure
                logger.error(f"DialogManager returned unexpected/invalid result for {conversation_id}: {dm_result}")
                response_text = "I'm sorry, I encountered an internal processing issue. Could you please try again?"

            # Ensure response_text is always a string
            if not isinstance(response_text, str):
                logger.error(f"Generated response is not a string for {conversation_id}: {type(response_text)}")
                response_text = "Error: Invalid response format."

            return DialogResponse(text=response_text, conversation_id=conversation_id)

        except Exception as e:
            # Log the full exception details for server-side debugging
            logger.error(f"Unhandled error processing dialog turn for {conversation_id}: {e}", exc_info=True)
            # Return a generic, user-friendly error response
            # Avoid exposing internal error details to the client
            return DialogResponse(
                text="I apologize, but an unexpected internal error occurred. Please try again later.",
                conversation_id=conversation_id
            )
    ```

  - **Verification:** Display the _entire modified `api.py`_ file. Manually review:
    - Is the `TowingState` class gone?
    - Is the old hardcoded `if/elif/else` logic in `process_dialog` gone?
    - Is the NLU/DM instantiation correct at the top level?
    - Does the `process_dialog` function body match the new implementation, primarily calling `dialog_manager.process_turn` and handling its result?
    - Are the error handling `try...except` blocks present?

- **Confirmation:** State "Phase 2 completed successfully. All actions performed and verified."

**Phase 3: Integration Testing**

- **Goal:** Verify the integrated system works correctly via API calls.

- **Step 3.1: Create API Integration Test File**

  - **Action:** Create a new file `test_api_integration.py`. Add the following structure:

    ```python
    # test_api_integration.py
    import pytest
    import httpx # Async HTTP client compatible with FastAPI/asyncio
    import asyncio

    # Base URL for the running API
    # Assumes API runs on localhost:8000. Adjust if needed.
    API_URL = "http://127.0.0.1:8000"
    DIALOG_ENDPOINT = f"{API_URL}/api/dialog"

    # --- Test Helper ---
    async def send_dialog_message(client: httpx.AsyncClient, text: str, conv_id: str = None) -> httpx.Response:
        """Helper function to send a message to the dialog endpoint."""
        payload = {"text": text}
        if conv_id:
            payload["conversation_id"] = conv_id
        try:
            response = await client.post(DIALOG_ENDPOINT, json=payload, timeout=10) # Add timeout
            response.raise_for_status() # Raise exception for 4xx/5xx responses
            return response
        except httpx.RequestError as exc:
            pytest.fail(f"HTTP Request failed: {exc}")
        except httpx.HTTPStatusError as exc:
             pytest.fail(f"HTTP Error {exc.response.status_code} - {exc.response.text}")

    # --- Test Cases ---
    @pytest.mark.asyncio
    async def test_api_towing_flow():
        """Test a multi-turn towing conversation via the API."""
        async with httpx.AsyncClient() as client:
            conv_id = None # Let the first request generate the ID

            # 1. Initial Tow Request
            response = await send_dialog_message(client, "My car broke down, I think I need a tow")
            data = response.json()
            conv_id = data["conversation_id"] # Store conversation ID
            assert "location" in data["text"].lower() # Should ask for location
            assert conv_id is not None

            # 2. Provide Location
            response = await send_dialog_message(client, "I'm at 555 Garage Street", conv_id)
            data = response.json()
            assert "destination" in data["text"].lower() # Should ask for destination

            # 3. Provide Destination
            response = await send_dialog_message(client, "Tow it to Mike's Auto Repair", conv_id)
            data = response.json()
            assert "vehicle" in data["text"].lower() and "make" in data["text"].lower() # Should ask for vehicle make

            # 4. Provide Vehicle Info (Example: Assuming DM asks make->model->year)
            response = await send_dialog_message(client, "It's a Ford", conv_id)
            data = response.json()
            assert "model" in data["text"].lower()
            response = await send_dialog_message(client, "F-150", conv_id)
            data = response.json()
            assert "year" in data["text"].lower()
            response = await send_dialog_message(client, "2021", conv_id)
            data = response.json()
            assert "confirm" in data["text"].lower() # Should ask for confirmation
            assert "ford" in data["text"].lower()
            assert "f-150" in data["text"].lower()
            assert "2021" in data["text"].lower()
            assert "555 garage street" in data["text"].lower()
            assert "mike's auto repair" in data["text"].lower()

            # 5. Confirm
            response = await send_dialog_message(client, "Yes confirm", conv_id)
            data = response.json()
            assert any(word in data["text"].lower() for word in ["confirmed", "booked", "dispatched", "complete", "all set"]) # Should confirm completion

    @pytest.mark.asyncio
    async def test_api_roadside_flow():
        """Test a simple roadside assistance flow."""
        async with httpx.AsyncClient() as client:
            conv_id = None
            response = await send_dialog_message(client, "My battery died")
            data = response.json()
            conv_id = data["conversation_id"]
            assert "location" in data["text"].lower() # Ask location

            response = await send_dialog_message(client, "At the library parking lot", conv_id)
            data = response.json()
             # Depending on slots defined, might ask vehicle or go to confirm
            assert "vehicle" in data["text"].lower() or "confirm" in data["text"].lower()

            # Add more steps if needed (vehicle info) then confirm
            # ...

            # Assuming it goes to confirmation now
            response = await send_dialog_message(client, "Yes", conv_id)
            data = response.json()
            assert any(word in data["text"].lower() for word in ["confirmed", "booked", "dispatched", "complete", "all set", "technician"])

    @pytest.mark.asyncio
    async def test_api_fallback():
         """Test fallback for an out-of-scope request."""
         async with httpx.AsyncClient() as client:
            response = await send_dialog_message(client, "What's the weather like?")
            data = response.json()
            # Check for fallback keywords
            assert any(word in data["text"].lower() for word in ["sorry", "understand", "towing", "roadside", "appointment", "assist", "clarify"])

    @pytest.mark.asyncio
    async def test_api_state_persistence():
        """Test that conversation state is maintained."""
        async with httpx.AsyncClient() as client:
            # Conversation 1
            response1_turn1 = await send_dialog_message(client, "Need a tow")
            data1_t1 = response1_turn1.json()
            conv_id_1 = data1_t1["conversation_id"]
            assert "location" in data1_t1["text"].lower()

            # Conversation 2 (Different ID)
            response2_turn1 = await send_dialog_message(client, "Need an oil change appointment")
            data2_t1 = response2_turn1.json()
            conv_id_2 = data2_t1["conversation_id"]
            assert "location" in data2_t1["text"].lower() or "service type" in data2_t1["text"].lower() # Appointment asks different things
            assert conv_id_1 != conv_id_2 # Ensure different IDs

            # Conversation 1 - Turn 2
            response1_turn2 = await send_dialog_message(client, "At the mall", conv_id_1)
            data1_t2 = response1_turn2.json()
            assert "destination" in data1_t2["text"].lower() # Should remember it's a tow flow

            # Conversation 2 - Turn 2
            response2_turn2 = await send_dialog_message(client, "Downtown service center", conv_id_2)
            data2_t2 = response2_turn2.json()
            assert "vehicle" in data2_t2["text"].lower() # Should remember it's an appointment flow
    ```

  - **Verification:** Confirm the file `test_api_integration.py` exists and contains the test structure above.

- **Step 3.2: Install Test Dependencies**

  - **Action:** Execute `pip install pytest httpx`
  - **Verification:** Command completes without errors.

- **Step 3.3: Run API Integration Tests**

  - **Action:**
    1.  Start the API server in one terminal: `python api.py`
    2.  In _another_ terminal (with the venv activated), run: `pytest test_api_integration.py`
  - **Verification:** All tests in `test_api_integration.py` MUST pass. If any fail, analyze the failure trace. The issue could be in `api.py`'s handling, `DialogManager`'s logic, `ResponseGenerator`, or `NLUInferencer`. Debug according to Rule 6, focusing on the refactored code. STOP if tests fail after one fix attempt. Remember to stop the API server after testing.

- **Step 3.4: Verify Core NLU Test**

  - **Action:** Execute `python test_integration.py`
  - **Verification:** Test MUST pass. If it fails, it indicates an unexpected issue with the core NLU loading or prediction, likely unrelated to the dialog refactor but needs stopping.

- **Confirmation:** State "Phase 3 completed successfully. All actions performed and verified."

**Phase 4: Cleanup and Documentation**

- **Goal:** Finalize requirements and documentation.

- **Step 4.1: Update Requirements Files**

  - **Action:**
    1.  Open `requirements-api.txt`. Review the dependencies. Ensure only packages needed to _run_ `api.py`, `inference.py`, `dialog_manager.py`, `response_generator.py` are present. Typically: `fastapi`, `uvicorn`, `pydantic`, `python-multipart` (if using form data, maybe not needed here), `transformers`, `torch`, `numpy`. Remove `datasets`, `scikit-learn`, `seqeval` if they aren't directly imported by the runtime code.
    2.  Open `requirements.txt`. Ensure it contains _all_ dependencies needed for training and testing, including those removed from `-api.txt` plus `pytest`, `httpx`.
  - **Verification:** Display the final content of _both_ `requirements-api.txt` and `requirements.txt`. Manually confirm the separation of runtime vs. development/training dependencies.

- **Step 4.2: Update Documentation**

  - **Action:**
    1.  Open `README.md`. Update the architecture description to reflect the unified `DialogManager` approach. Remove mentions of the split logic.
    2.  Open `API_README.md`. Update the description of `/api/dialog` to emphasize its delegation role. Remove mentions of internal `TowingState`.
  - **Verification:** Manually confirm the documentation accurately reflects the refactored structure.

- **Step 4.3: Code Formatting**

  - **Action:** Execute `black .`
  - **Verification:** Command completes without errors. Check `git status` to see which files were formatted.

- **Confirmation:** State "Phase 4 completed successfully. All actions performed and verified."

**Phase 5: Final Verification**

- **Goal:** Ensure everything is stable and ready.

- **Step 5.1: Run All Tests**

  - **Action:** Execute the following commands sequentially:
    1.  `python -m unittest test_dialog_manager_unified.py`
    2.  `python test_integration.py`
    3.  Start API: `python api.py`
    4.  In another terminal: `pytest test_api_integration.py`
    5.  Stop the API server (Ctrl+C).
  - **Verification:** All test suites MUST pass. If any fail, STOP and report (Rule 6).

- **Step 5.2: Final Review**

  - **Action:** Execute `git diff`.
  - **Verification:** Review the collected changes. Do they align with the plan? Are there any unexpected modifications? Report completion.

- **Confirmation:** State "Phase 5 completed successfully. Final verification passed. Refactoring complete."

---

This plan is intentionally verbose and includes strict verification and halting conditions. Follow it precisely. Good luck, Cursor!
