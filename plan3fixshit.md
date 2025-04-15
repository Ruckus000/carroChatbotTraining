Okay, let's create an extremely detailed, locked-down plan for Cursor to address the remaining issues and finalize the simplified NLU system. The emphasis is on sequential execution, strict adherence, verification, and avoiding deviation.

**Goal:** Produce a clean, functional, simplified NLU system consisting of `data/nlu_training_data.json`, `train.py`, `inference.py` (with only `NLUInferencer`), necessary support files (`requirements.txt`, `README.md`, `.gitignore`), passing phase tests, and a verified `test_integration.py` running against _real_, successfully trained models located in `./trained_nlu_model/`. All other code related to LangGraph, context awareness, old training pipelines, etc., MUST be removed.

**General Instructions for Cursor:**

- **Execute Sequentially:** You MUST complete all steps and pass the test for a given Phase before starting the next Phase. NO EXCEPTIONS.
- **Strict Adherence:** Execute ONLY the commands and code modifications specified. DO NOT add any extra code, optimizations, files, or features. DO NOT modify any files not explicitly mentioned in a step.
- **CPU Only:** All operations MUST use the CPU. Verify `no_cuda=True` is set in `TrainingArguments` in `train.py`.
- **Test Integrity:** You MUST execute the specified test scripts exactly as provided. You MUST NOT modify the test scripts (`test_phase*.py`, `test_integration.py`). If a test fails, you MUST modify the code generated or manipulated _within that phase_ to fix the issue and then rerun the test until it passes.
- **Error Reporting:** If any command fails, or if a test fails and you cannot identify the cause in the code _you were instructed to modify in that phase_, STOP execution and report the exact command/step that failed and the full error message.
- **File Paths:** All file paths are relative to the project root directory.

---

**Phase 1: Ensure Training Success and Artifact Generation**

**Goal:** Execute `train.py` successfully, fix known issues, and verify that the correct model artifacts are created in the `./trained_nlu_model/` directory.

**Steps:**

1.  **Confirm Target Script:** Verify that the file `train.py` exists in the root directory.
2.  **Edit `train.py` for Known Fixes:**
    - Open `train.py`.
    - Locate the two instances of `TrainingArguments(...)`.
    - Inside _both_ instances, find the parameter `evaluation_strategy`. **Change this parameter name** to `eval_strategy`. Ensure its value is `"epoch"`.
    - Inside _both_ instances, find the parameter `save_strategy`. Ensure its value is `"epoch"`.
    - Inside _both_ instances, verify `no_cuda=True` is present and set to `True`.
    - Locate the `prepare_entity_dataset` function. Find the loop where it processes `examples['text']` and `examples['entities']`. Ensure the entire processing for a single example (inside the `for i, (text, entities) ...` loop) is wrapped in a `try...except Exception as e:` block. Inside the `except` block, add `print(f"Error preparing entity example {i} for text '{text[:50]}...': {e}")` and `continue` to skip the problematic example without stopping the entire process.
    - Locate the section preparing `val_intent_ids`. Find the `if intent not in intent2id:` block. Ensure the warning message `print(f"Warning: Unseen intent '{intent}'...")` is present. Ensure it correctly assigns a valid fallback ID (e.g., `fallback_id = intent2id.get('fallback_out_of_scope', 0)` or similar logic already present) instead of crashing.
    - Save the changes to `train.py`.
3.  **Clean Previous Output:** Execute the following command in the terminal from the project root to remove any previous potentially incomplete model outputs:
    ```bash
    python -c "import shutil; shutil.rmtree('./trained_nlu_model', ignore_errors=True); print('Cleaned ./trained_nlu_model/ directory.')"
    ```
4.  **Execute Training Script:** Run the training script from the project root:
    ```bash
    python train.py
    ```
5.  **Monitor Execution:** Observe the console output. Note any errors or warnings, especially related to data preparation or unseen labels. The script MUST complete without crashing.
6.  **Verify Output Artifacts:** After `train.py` finishes, verify the existence and basic contents of the output:
    - Check if the directory `./trained_nlu_model/` exists.
    - Check if the subdirectory `./trained_nlu_model/intent_model/` exists.
    - Check if the subdirectory `./trained_nlu_model/entity_model/` exists.
    - Verify the presence of the following files within `./trained_nlu_model/intent_model/`: `config.json`, (`pytorch_model.bin` OR `model.safetensors`), `tokenizer_config.json`, `vocab.txt`, `intent2id.json`.
    - Verify the presence of the following files within `./trained_nlu_model/entity_model/`: `config.json`, (`pytorch_model.bin` OR `model.safetensors`), `tokenizer_config.json`, `vocab.txt`, `tag2id.json`.

**Phase 1 Test:**

- **Instruction:** Run the test script for Phase 2 artifacts: `python test_phase2.py`. **DO NOT MODIFY `test_phase2.py`**.
- **Debugging:** If the test fails:
  - Compare the actual file structure in `./trained_nlu_model/` against the requirements in Step 6.
  - Review the console output from `python train.py` (Step 4) for errors that might explain missing files (e.g., errors during saving).
  - Review `train.py` again, focusing on the `save_pretrained` calls and the output directory paths specified in `TrainingArguments`.
  - Repeat Steps 3-6 until `test_phase2.py` passes. Report success or failure.

---

**Phase 2: Clean Inference Script and Verify Structure**

**Goal:** Modify `inference.py` to remove all code unrelated to the simple `NLUInferencer` class and ensure the remaining class structure is correct.

**Steps:**

1.  **Locate `inference.py`:** Confirm the `inference.py` script exists in the root directory.
2.  **Edit `inference.py`:**
    - Open the file `inference.py`.
    - **Delete** the entire class definition for `CarroAssistant`.
    - **Delete** the entire class definition for `ContextAwareCarroAssistant`.
    - Review the import statements at the top of the file. **Delete** any imports that are no longer used after removing the above classes (e.g., `difflib`, `uuid`, potentially others if they were only for the deleted classes). Keep `os`, `json`, `torch`, `numpy`, and the specific `transformers` imports needed by `NLUInferencer`.
    - Verify that the `NLUInferencer` class is the only class remaining in the file.
    - Verify the `__init__` method loads models and tokenizers from `./trained_nlu_model/intent_model` and `./trained_nlu_model/entity_model` respectively.
    - Verify the `predict` method calls `_predict_intent` and `_predict_entities` and returns the dictionary in the correct format.
    - Save the changes to `inference.py`.

**Phase 2 Test:**

- **Instruction:** Run the test script for Phase 3 structure: `python test_phase3.py`. **DO NOT MODIFY `test_phase3.py`**.
- **Debugging:** If the test fails:
  - Check for `ImportError`, `NameError` or `AttributeError` caused by the code deletion in Step 2.
  - Ensure the `NLUInferencer` class structure, `__init__` method, and `predict` method signature and return format match exactly what `test_phase3.py` expects.
  - Review and fix `inference.py`. Repeat until `test_phase3.py` passes. Report success or failure.

---

**Phase 3: Test Integration with Real Models**

**Goal:** Verify that the `NLUInferencer` (cleaned in Phase 2) can successfully load and use the models produced by `train.py` (verified in Phase 1) to make plausible predictions on sample inputs.

**Steps:**

1.  **Locate `test_integration.py`:** Confirm the **root-level** `test_integration.py` script exists.
2.  **Edit `test_integration.py`:**
    - Open the file `test_integration.py`.
    - Locate and **DELETE** or **COMMENT OUT** all lines of code related to mocking file existence or patching the `transformers` library. Specifically, remove the section marked `# Create directory mocking...` down to `# ---- End Mocking ----` and any lines using `patch(...)`. The script must now import and use the real `NLUInferencer` which loads the actual models.
    - Ensure the script still imports `from inference import NLUInferencer`.
    - Save the changes to `test_integration.py`.
3.  **Execute Integration Test:** Run the script from the project root:
    ```bash
    python test_integration.py
    ```
4.  **Analyze Output:** The script will print `[PASS]` or `[FAIL]` for the intent type check and `[WARN]` for missing expected entity types. The overall test passes only if _all_ intent checks pass.

**Phase 3 Test:**

- **Instruction:** The execution of `python test_integration.py` _is_ the test. The script must report "All tests PASSED!".
- **Debugging:** If the test fails (`[FAIL]` messages or errors):
  - **Model Loading Errors:** Check `NLUInferencer.__init__` in `inference.py`. Are the paths `./trained_nlu_model/intent_model` and `./trained_nlu_model/entity_model` correct? Did Phase 1 definitely complete successfully?
  - **Prediction Errors:** Look at the specific test case that failed.
    - _Intent Mismatch:_ Check the confidence score printed. If it's low (<0.5), the `fallback_low_confidence` assignment is correct. If the confidence is high but the intent is wrong, the intent model might need more/better training data (note this, but don't retrain yet).
    - _Entity Errors:_ Carefully debug the `_predict_entities` method in `inference.py`, focusing on the BIO tag grouping logic. Add print statements to see the `word_predictions` list and how it's being converted into the final `entities` list. Check the `word_ids` alignment.
  - Review and fix `inference.py`. If absolutely necessary after confirming inference logic is sound, consider rerunning Phase 1 (`python train.py`) perhaps with one more epoch, then repeat Phase 3.
  - Repeat Step 3 until `test_integration.py` passes. Report success or failure.

---

**Phase 4: Final Project Cleanup**

**Goal:** Remove all remaining obsolete files and directories, leaving only the core simplified NLU system components. Update `requirements.txt` and `README.md`.

**Steps:**

1.  **Locate `cleanup.py`:** Confirm the `cleanup.py` script exists.
2.  **Review `cleanup.py`:** Ensure the lists `files_to_delete` and `dirs_to_delete` accurately reflect **all** obsolete items identified in the previous plan's Phase 5 (including old data files, old scripts, context files, LangGraph files, etc.). Ensure the `files_to_preserve` list _includes_ `test_phase5.py` and `cleanup.py` itself, in addition to the core NLU files. Verify the `requirements.txt` content and `README.md` content within the script match the simplified system.
3.  **Execute Cleanup:** Run the script from the project root:
    ```bash
    python cleanup.py
    ```
4.  **Review CI Workflow:**
    - Open `.github/workflows/ci.yml`.
    - **Remove** any steps or commands that reference deleted files or directories (e.g., tests in `tests/` that were deleted, linting/testing steps for `langgraph_integration`).
    - Ensure the remaining steps correctly install dependencies from the cleaned `requirements.txt` and run the remaining tests (e.g., root `test_integration.py`).
    - Save the changes to `.github/workflows/ci.yml`.

**Phase 4 Test:**

- **Instruction:** Run the test script for Phase 5 cleanup: `python test_phase5.py`. **DO NOT MODIFY `test_phase5.py`**.
- **Debugging:** If the test fails:
  - Check the error messages indicating which files/directories were expected to be deleted but weren't, or which required files are missing.
  - Manually delete any remaining obsolete files/folders.
  - Ensure `requirements.txt` and `README.md` were correctly generated by `cleanup.py`.
  - If necessary, fix `cleanup.py` and rerun it (Step 2 & 3).
  - Repeat until `test_phase5.py` passes. Report success or failure.

---

**Final Confirmation:**

- After completing all phases and passing all tests, confirm: "All phases completed successfully. The simplified NLU system is trained, verified via integration testing with real models, and the project directory is cleaned."
