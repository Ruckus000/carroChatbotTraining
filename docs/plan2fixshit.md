Okay, let's refine the plan with more explicit instructions, especially concerning BIO tag handling and basic error handling, to minimize the chances of Cursor misinterpreting or failing.

General Instructions for Cursor: (Remain the same)

Follow Phases Sequentially: Complete all steps within a phase before moving to the next.

Adhere Strictly to Instructions: Do not add extra features, models, or complexity beyond what is explicitly asked for in each phase. The goal is simplification.

Do Not Modify Tests: Execute the provided test scripts after completing the implementation steps for each phase. If a test fails, modify the code you generated in that phase to fix the issue. DO NOT CHANGE THE TEST SCRIPT ITSELF. If you believe a test is fundamentally wrong, stop and report the issue.

Code Placement: Place new scripts (train.py, inference.py, test scripts) directly inside the root directory of the project for simplicity initially, unless specified otherwise. We can reorganize later if needed.

Error Reporting: If you encounter errors you cannot resolve or instructions are unclear, stop and ask for clarification.

Use Basic Error Handling: Wrap file operations (reading/writing JSON, checking paths) and model loading/training steps in basic try...except blocks to catch common errors like FileNotFoundError or JSONDecodeError and print informative error messages.

Phase 1: Data Consolidation and Preparation [COMPLETED]

Goal: Create a single, unified training data file in a format suitable for joint intent and entity recognition, removing the old complex data structure.

Steps:

Create New Data File: Create a new file named data/nlu_training_data.json.

Load Source Data:

Implement a try...except FileNotFoundError block when opening data/sample_conversations.json.

Implement a try...except json.JSONDecodeError block when loading the JSON data. Print informative errors if exceptions occur.

Read the contents of the original data/sample_conversations.json file.

Transform and Consolidate:

Iterate through each example loaded in the previous step.

For each example, create a new dictionary with the following keys:

text: Copy the input value. Ensure it's a string.

intent: Create a flattened intent string. Combine flow and intent using an underscore (e.g., towing_request_tow_basic, roadside_request_roadside_battery, appointment_book_service_type, fallback_out_of_domain, clarification_ambiguous_request). Ensure the source flow and intent keys exist before accessing them.

entities: Copy the entities list as is from the source. Ensure it's a list, even if empty. Each item in the list must be a dictionary containing entity (string) and value (usually string) keys.

Important: Do not include examples from data/augmented_sample_conversations.json or data/combined_conversations.json.

Add a few examples manually representing fallbacks if they aren't naturally covered, using an appropriate flattened intent like fallback_out_of_scope_weather. Example:

{
"text": "What's the weather like?",
"intent": "fallback_out_of_scope_weather",
"entities": []
}

Save New Data:

Implement a try...except IOError block for writing the file.

Write the consolidated list of transformed dictionaries to data/nlu_training_data.json in JSON format using json.dump() with indent=2 for readability and ensure_ascii=False.

Verify Output: Briefly inspect the data/nlu_training_data.json file to ensure it contains the expected structure (text, intent, entities) and flattened intents.

Phase 1 Test: (Test script remains the same as previously provided)

Instruction: Create a new file named test_phase1.py. Paste the code provided in the previous response into it. DO NOT MODIFY THIS TEST SCRIPT. Run it using python test_phase1.py. If it fails, fix the data/nlu_training_data.json file generated in the steps above until the test passes.

Phase 2: Simplified Training Implementation [COMPLETED]

Goal: Create a training script (train.py) that uses Hugging Face Transformers to train two separate models (one for intent, one for entities) based on the consolidated data.

Steps:

Create Training Script: Create a new file named train.py.

Import Libraries: Import necessary libraries (json, os, transformers, torch, datasets, sklearn.model_selection, sklearn.metrics, numpy, seqeval).

Load Data Function: Create a function load_data(filepath) that loads the JSON data from data/nlu_training_data.json. Include try...except blocks for FileNotFoundError and json.JSONDecodeError.

Split Data: Use sklearn.model_selection.train_test_split to split the loaded data into training (e.g., 80%) and validation (e.g., 20%) sets. Use a fixed random_state for reproducibility.

Prepare Intent Data:

Create mappings for unique intents to integer IDs (intent2id, id2intent) from the training data.

Convert training and validation data into lists of texts and corresponding intent IDs. Handle potential unseen intents in the validation set gracefully (e.g., map them to a default ID or log a warning).

Tokenize the texts using DistilBertTokenizer.from_pretrained('distilbert-base-uncased') with padding and truncation.

Create a custom Hugging Face Dataset class or use datasets.Dataset.from_dict for intent training data (input_ids, attention_mask, labels where labels are the intent IDs).

Prepare Entity Data (Explicit BIO Alignment):

Get All Tags: Determine the set of unique entity tags (e.g., pickup_location, vehicle_make) from the training data. Create the full BIO tag set (e.g., B-pickup_location, I-pickup_location, etc., plus O).

Create Tag Mappings: Create mappings for these BIO tags to integer IDs (tag2id, id2tag). Ensure O maps to 0.

Tokenize and Align Function: Create a function tokenize_and_align_labels(examples, tokenizer, tag2id):

Takes a batch of examples (each having text and entities).

Tokenizes the text using tokenizer(..., truncation=True, padding=True, is_split_into_words=False). Crucially, use is_split_into_words=False here.

Get word_ids from the tokenizer output for each example in the batch.

For each example, initialize labels list with -100.

Iterate through the word_ids. If word_ids[i] is not None (it's part of an original word):

Find the corresponding (word, bio_tag) from the pre-calculated BIO tags for the original text (you'll need a helper function convert_text_entities_to_bio(text, entities) that splits the text and generates BIO tags before this tokenization step).

Get the integer ID for the bio_tag using tag2id.

Set labels[i] to this ID. If it's the first token of a word (check word_ids[i] != word_ids[i-1]), use the tag. For subsequent tokens of the same word, set labels[i] = -100 (as per standard token classification practice, unless your strategy differs - -100 is common).

Return the tokenized inputs along with the aligned labels.

Apply this function to your training and validation datasets using dataset.map(..., batched=True).

Load Models:

Inside a try...except block, load DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(intent2id), id2label=id2intent, label2id=intent2id).

Inside a try...except block, load DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(tag2id), id2label=id2tag, label2id=tag2id).

Configure Training:

Set up TrainingArguments for both intent and entity trainers. Use small epochs (1-2), small batch sizes (4 or 8), specify distinct output directories (./trained_nlu_model/intent_model_checkpoints, ./trained_nlu_model/entity_model_checkpoints), logging_steps, evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True. Force CPU: Add no_cuda=True. Use metric_for_best_model='f1' for entities and 'accuracy' or 'f1' for intent.

Define compute_metrics function for intent: uses sklearn.metrics.accuracy_score, precision_score, recall_score, f1_score (use average='weighted' or 'macro').

Define compute_metrics function for entities: uses seqeval.metrics.classification_report or calculates precision/recall/F1 based on aligned labels (ignoring -100).

Train Models:

Create a Trainer instance for the Intent Model. Include appropriate Dataset and compute_metrics.

Wrap the trainer.train() call in a try...except block.

Create a Trainer instance for the Entity Model. Include appropriate Dataset and compute_metrics.

Wrap the trainer.train() call in a try...except block.

Save Models:

Inside try...except blocks:

Save the fine-tuned Intent Model and its tokenizer to ./trained_nlu_model/intent_model.

Save the fine-tuned Entity Model and its tokenizer to ./trained_nlu_model/entity_model.

Save the intent2id and tag2id mappings as JSON files in their respective model directories using json.dump() with error handling.

Phase 2 Test: (Test script remains the same as previously provided)

Instruction: Create a new file named test_phase2.py. Paste the code provided in the previous response into it. DO NOT MODIFY THIS TEST SCRIPT. Run python train.py first. Then run python test_phase2.py. If it fails, fix train.py and rerun training, then re-test until test_phase2.py passes.

Phase 3: Simplified Inference Implementation

Goal: Create an inference script/class (inference.py) that loads the trained models (intent and entity) and predicts intent/entities for a given text input.

Steps:

Create Inference Script: Create a new file named inference.py.

Import Libraries: Add torch, transformers, json, os, numpy.

Define Inference Class: Create a class (e.g., NLUInferencer).

Initialization (**init**):

Takes the model base path (./trained_nlu_model) as input.

Use try...except blocks for loading each model component (model, tokenizer, label maps).

Loads the Intent Model (DistilBertForSequenceClassification) and its tokenizer from ./trained_nlu_model/intent_model.

Loads the intent2id.json and creates id2intent.

Loads the Entity Model (DistilBertForTokenClassification) and its tokenizer from ./trained_nlu_model/entity_model.

Loads the tag2id.json and creates id2tag.

Sets the device to CPU (torch.device("cpu")) and moves models to CPU (.to(device)). Sets models to evaluation mode (.eval()).

Prediction Method (predict):

Takes text (string) as input.

Wrap the entire prediction logic in a try...except block to catch runtime errors.

Predict Intent:

Tokenize the text using the intent tokenizer (padding=True, truncation=True, return_tensors="pt"). Move tensors to the CPU device.

Use with torch.no_grad(): for inference.

Pass inputs to the intent model.

Get logits, calculate probabilities (torch.softmax).

Find the highest probability intent ID (torch.argmax) and its confidence score.

Convert intent ID back to the intent label string using id2intent.

Predict Entities (Explicit Grouping Logic):

Tokenize the text using the entity tokenizer. Crucially, ensure you handle subword token alignment correctly. Use tokenizer(text.split(), truncation=True, padding=True, is_split_into_words=True, return_tensors="pt"). Get word_ids(). Move tensors to CPU.

Use with torch.no_grad(): for inference.

Pass inputs to the entity model.

Get the highest probability tag ID for each token using torch.argmax. Convert predictions and word_ids to lists/numpy arrays.

Align Predictions to Words: Create a list of (word, predicted_tag) pairs. Iterate through tokens and predictions. Use word_ids to map token predictions back to the original words. For words split into multiple tokens, typically use the prediction of the first token associated with that word. Ignore predictions for special tokens (where word_id is None). Use id2tag to get tag names.

Group BIO Tags: Initialize extracted_entities_list = [], current_entity_tokens = [], current_entity_type = None. Iterate through the (word, predicted_tag) pairs:

If predicted_tag starts with B-:

If current_entity_type is not None (we were processing a previous entity), join current_entity_tokens into a string and add {'entity': current_entity_type, 'value': joined_string} to extracted_entities_list.

Start a new entity: current_entity_tokens = [word], current_entity_type = predicted_tag[2:].

If predicted_tag starts with I-:

If current_entity_type is None or predicted_tag[2:] != current_entity_type (misaligned I tag), treat it as O (or discard the token for entity purposes). Optionally log a warning.

Else (aligned I tag): append word to current_entity_tokens.

If predicted_tag == 'O':

If current_entity_type is not None, finalize the entity: join current_entity_tokens, add the dictionary to extracted_entities_list, and reset current_entity_tokens = [], current_entity_type = None.

After Loop: If current_entity_type is still active, finalize the last entity.

Basic Fallback Logic:

Define a confidence threshold (e.g., CONFIDENCE_THRESHOLD = 0.5).

If predicted_intent_confidence < CONFIDENCE_THRESHOLD, set predicted_intent_label = 'fallback_low_confidence'.

Return Value: Return the dictionary as specified before: {"text": ..., "intent": {"name": ..., "confidence": ...}, "entities": ...}.

Phase 3 Test: (Test script remains the same as previously provided, including the mocking setup)

Instruction: Create a new file named test_phase3.py. Paste the code provided in the previous response into it. DO NOT MODIFY THIS TEST SCRIPT. Run python test_phase3.py. If it fails, fix inference.py until the test passes.

Phase 4: Integration and Basic Testing [COMPLETED]

Goal: Create a minimal test script (test_integration.py) that uses the NLUInferencer to process a few sample inputs and checks if the output intent/entities are plausible.

Steps:

Create Test Harness: Create a new file named test_integration.py.

Import: Import the NLUInferencer class from inference.py and os, json (for mocking).

Add Mocking: Include the file/directory mocking code from test_phase3.py at the beginning of this script to ensure it can run even if Phase 2 wasn't fully successful.

Initialize: Inside a try...except block, create an instance of NLUInferencer.

Define Test Cases: Define the list of test cases as specified in the previous plan (basic tow, roadside, appointment, out-of-scope, ambiguous).

Run Predictions: Loop through the test cases. For each case:

Wrap the call to inferencer.predict() inside a try...except block.

Print the input text and the prediction result.

Perform the intent and basic entity type assertions as specified previously. Handle potential KeyError if the prediction structure is wrong.

Report Summary: Print the summary.

Phase 4 Test: (Test script remains the same as previously provided)

Instruction: Create a new file named test_integration.py. Paste the code provided in the previous response into it. DO NOT MODIFY THIS TEST SCRIPT. Run python test_integration.py. If it fails, investigate inference.py or potentially train.py.

Phase 5: Cleanup and Documentation

Goal: Remove obsolete files and update the main README.md to reflect the simplified structure and usage.

Steps:

Delete Obsolete Files/Folders: Delete the files/folders listed in the previous plan. Use basic try...except OSError blocks around os.remove() or os.rmdir() / shutil.rmtree() calls for robustness. Be very careful with deletions.

Update requirements.txt: Review and trim requirements.txt as specified previously.

Update README.md: Create or replace the main README.md file with the content described previously (setup, data format, training command, inference usage example, testing).

Phase 5 Test: (Test script remains the same as previously provided)

Instruction: Create a new file named test_phase5.py. Paste the code provided in the previous response into it. DO NOT MODIFY THIS TEST SCRIPT. Run python test_phase5.py. If it fails, ensure the correct files were deleted and the main README.md exists.

These refinements add explicit instructions for the tricky parts (BIO alignment/grouping) and basic error handling, making the plan more robust for AI execution while still achieving the simplification goal.

Phase 6: Recovery and Proper Project Structure

Goal: Fix the issues created in Phase 5 where essential implementation files were mistakenly deleted. Restore the core functionality while maintaining the simplified structure.

Problem Assessment:

- Phase 5 implementation deleted ALL files rather than just obsolete ones
- Essential implementation files (`train.py`, `inference.py`, `test_*.py`) are missing
- Data file (`nlu_training_data.json`) is missing
- Only the directory structure, README.md, and requirements.txt remain

Steps:

1. Recover Implementation Files:

   a. Restore `train.py` (from Phase 2):

   - Retrieve from git history or recreate the Phase 2 implementation
   - Place it in the root directory
   - Ensure it loads data from `data/nlu_training_data.json`
   - Ensure it saves models to the `trained_nlu_model` directories
   - Include proper error handling for file operations and model loading

   b. Restore `inference.py` (from Phase 3):

   - Retrieve from git history or recreate the Phase 3 implementation
   - Place it in the root directory
   - Ensure it properly implements the `NLUInferencer` class with methods as specified in Phase 3
   - Confirm proper error handling for model loading and inference

   c. Restore Test Files:

   - Restore `test_phase1.py`, `test_phase2.py`, `test_phase3.py`, and `test_integration.py`
   - Place them in the root directory
   - Do not modify the test scripts, as per the original instructions

2. Restore Data File:

   a. Create Sample Training Data:

   - Create the `data/nlu_training_data.json` file with representative examples
   - Include examples for various intents (towing, roadside, appointment)
   - Include examples with entities (locations, vehicle details)
   - Include fallback examples
   - Follow the specified format with text, intent, and entities fields

3. Verify Consistency:

   a. Update `test_phase5.py`:

   - Modify it to check for all essential files that should be present
   - Include checks for `train.py`, `inference.py`, test files, and data file
   - Run it to verify the complete structure is in place

   b. Run Tests Sequentially:

   - Run `test_phase1.py` to verify data structure
   - Run `train.py` and `test_phase2.py` to verify training
   - Run `test_phase3.py` to verify inference
   - Run `test_integration.py` to verify integration

4. Ensure Clean Project Structure:

   a. Verify Directory Structure:

   - Root directory should contain implementation files, test files, README.md, and requirements.txt
   - `data/` directory should contain `nlu_training_data.json`
   - `trained_nlu_model/` should have subdirectories for intent and entity models

   b. Remove Any Remaining Obsolete Files:

   - Double-check against the list of files required for Phases 1-4
   - Only remove files verified as obsolete (not created in Phases 1-4)

5. Documentation:

   a. Add Notes to README.md:

   - Document the recovery process
   - Clarify the project structure
   - Update the usage examples as needed

Phase 6 Test:

Instruction: Execute the following test commands in sequence to verify that the recovery is complete and the system works as expected:

```bash
# Verify project structure
python test_phase5.py

# Verify data preparation
python test_phase1.py

# Train models and verify training
python train.py
python test_phase2.py

# Verify inference
python test_phase3.py

# Verify integration
python test_integration.py
```

All tests should pass, confirming that the system has been recovered properly and is functioning as specified in Phases 1-4.
