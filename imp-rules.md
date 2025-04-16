# CURSOR EXECUTION RULES - Simple NLU Refactoring

**Objective:** Execute the provided multi-phase plan precisely to refactor the project into a simplified NLU system (Intent Classification + Entity Recognition) and clean up obsolete code.

**Your Primary Directive:** Follow the phased plan with extreme accuracy. Your goal is _simplification_ and _cleanup_ according to the plan, NOT adding features, optimizations, or unrelated code.

---

## I. General Principles

1.  **Strict Plan Adherence:** You MUST follow the phases and steps in the provided plan sequentially and exactly as written. Do not skip steps or phases.
2.  **Targeted Modifications ONLY:** Only modify or create files and code explicitly mentioned in the current step of the current phase. DO NOT modify any other files or code, especially `train.py` or `inference.py`, unless the current step specifically instructs you to do so.
3.  **No Unsolicited Additions:** DO NOT add any new features, functions, classes, error handling (beyond basic `try...except` for file/model loading as instructed), optimizations, or logic that is not explicitly part of the current step's instructions.
4.  **CPU Execution ONLY:** All Python scripts, especially `train.py` and `inference.py`, MUST be configured and run using the CPU only. Ensure `no_cuda=True` is set in `TrainingArguments` and models are loaded onto the CPU device (`torch.device("cpu")`).
5.  **Use Provided Code Verbatim:** If a code snippet is provided in a step, use it EXACTLY as written unless the instruction explicitly states to adapt it.

## II. Phased Execution and Verification

6.  **Complete Phases Sequentially:** Finish ALL steps within the current phase, including passing the specified test, before moving to the next phase.
7.  **Mandatory Verification:** Each phase ends with a "Phase X Test" section. You MUST execute the specified test script command exactly as provided.
8.  **Test Integrity - CRITICAL:** **DO NOT MODIFY THE TEST SCRIPTS (`test_phase*.py`, `test_integration.py`) UNDER ANY CIRCUMSTANCES.** These tests are the objective measure of whether you completed the phase correctly. Modifying them defeats their purpose.
9.  **Debugging Failing Tests:** If a test fails:
    - Review the error message provided by the test script.
    - Identify the code file(s) you were instructed to modify or create _within that specific phase_.
    - Modify ONLY that code to fix the specific error reported by the test.
    - Re-run the test script for that phase.
    - Repeat this debug cycle until the test passes.
    - If you cannot make the test pass by modifying ONLY the code related to the current phase's instructions, STOP and report the failure (see Rule 12).
10. **Verify File/Directory Creation:** If a step requires creating files or directories (e.g., `./trained_nlu_model/`), explicitly verify their existence and expected contents _after_ executing the relevant command, before proceeding to the phase test.

## III. File and Path Handling

11. **Use Correct Paths:** All file and directory paths mentioned in the plan are relative to the project's root directory. Use these paths precisely. Do not assume paths or use absolute paths.
12. **Careful Deletion (Phase 4):** During the cleanup phase, only delete files and directories EXPLICITLY listed in the plan for deletion. Double-check the lists before executing deletion commands.

## IV. Reporting and Error Handling

13. **Confirm Phase Completion:** After successfully completing all steps in a phase AND passing its associated test, explicitly state: "Phase [X] completed successfully and Test [test_script_name.py] passed."
14. **STOP on Error:** If any command fails, if a test fails and you cannot resolve it by fixing the code _from that phase_, or if instructions are unclear, STOP EXECUTION IMMEDIATELY.
15. **Report Errors Clearly:** When stopping due to an error (Rule 14), report the following:
    - The Phase number and Step number you were executing.
    - The exact command you tried to run (if applicable).
    - The FULL error message or test failure output.
    - A brief description of what you tried to do to fix it (if applicable).

---

**Acknowledge these rules before starting Phase 1.** Refer back to these rules if you are unsure about any instruction in the plan. Your adherence to these rules is critical for project success.
