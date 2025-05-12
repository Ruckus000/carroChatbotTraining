import json
import os
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    pipeline,
)


class NLUInferencer:
    def __init__(self, model_path="./trained_nlu_model"):
        """
        Initialize the NLU Inferencer.

        Args:
            model_path (str): Path to the directory containing the trained models.
        """
        self.model_path = model_path

        # Determine device: MPS for Apple Silicon, else CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            print(
                "INFO [NLUInferencer]: MPS device found. Inference will use Apple Silicon GPU."
            )
        else:
            self.device = torch.device("cpu")
            print("INFO [NLUInferencer]: MPS device not found. Inference will use CPU.")

        self.CONFIDENCE_THRESHOLD = 0.01

        # Load intent model
        try:
            self.intent_model_path = os.path.join(model_path, "intent_model")
            self.intent_model = DistilBertForSequenceClassification.from_pretrained(
                self.intent_model_path
            )
            self.intent_model.to(self.device)
            self.intent_model.eval()

            self.intent_tokenizer = DistilBertTokenizer.from_pretrained(
                self.intent_model_path
            )

            # Load intent mappings
            with open(os.path.join(self.intent_model_path, "intent2id.json"), "r") as f:
                self.intent2id = json.load(f)

            # Create id2intent mapping
            self.id2intent = {v: k for k, v in self.intent2id.items()}

        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load intent model: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse intent mappings: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading intent model: {e}")

        # Load entity model
        try:
            self.entity_model_path = os.path.join(model_path, "entity_model")
            self.entity_model = DistilBertForTokenClassification.from_pretrained(
                self.entity_model_path
            )
            self.entity_model.to(self.device)
            self.entity_model.eval()

            self.entity_tokenizer = DistilBertTokenizer.from_pretrained(
                self.entity_model_path
            )

            # Load entity tag mappings
            with open(os.path.join(self.entity_model_path, "tag2id.json"), "r") as f:
                self.tag2id = json.load(f)

            # Create id2tag mapping
            self.id2tag = {v: k for k, v in self.tag2id.items()}

        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load entity model: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse entity tag mappings: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading entity model: {e}")

        # Load sentiment analysis pipeline
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=(
                    0 if self.device.type == "cuda" or self.device.type == "mps" else -1
                ),
            )
            print(
                f"INFO [NLUInferencer]: Sentiment analysis model loaded successfully using {model_name}."
            )
        except Exception as e:
            print(
                f"WARNING [NLUInferencer]: Failed to load sentiment analysis model: {e}"
            )
            self.sentiment_pipeline = None

    def predict(self, text):
        """
        Predict the intent and entities for the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing the text, intent, entities, and sentiment.
        """
        try:
            # Initialize response
            result = {"text": text, "intent": {}, "entities": [], "sentiment": {}}

            # Predict intent
            intent_prediction = self._predict_intent(text)
            result["intent"] = intent_prediction

            # Predict entities
            entities = self._predict_entities(text)
            result["entities"] = entities

            # Predict sentiment
            sentiment = self._predict_sentiment(text)
            result["sentiment"] = sentiment

            return result

        except Exception as e:
            # Fallback logic for runtime errors
            return {
                "text": text,
                "intent": {"name": "fallback_runtime_error", "confidence": 1.0},
                "entities": [],
                "sentiment": {"label": "neutral", "score": 0.5},
            }

    def _predict_intent(self, text):
        """
        Predict the intent for the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing the intent name and confidence.
        """
        try:
            # Tokenize the input
            inputs = self.intent_tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.intent_model(**inputs)

            # Calculate probabilities using softmax
            logits = outputs.logits.cpu()
            probabilities = torch.softmax(logits, dim=1).numpy()[0]

            # Get the predicted intent
            predicted_intent_id = np.argmax(probabilities)
            predicted_intent_confidence = float(probabilities[predicted_intent_id])

            # Map back to intent name
            predicted_intent_name = self.id2intent.get(
                int(predicted_intent_id), "unknown"
            )

            # Apply confidence threshold for fallback
            if predicted_intent_confidence < self.CONFIDENCE_THRESHOLD:
                predicted_intent_name = "fallback_low_confidence"

            return {
                "name": predicted_intent_name,
                "confidence": predicted_intent_confidence,
            }

        except Exception as e:
            return {"name": "fallback_intent_error", "confidence": 1.0}

    def _predict_sentiment(self, text):
        """
        Predict the sentiment of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing sentiment label and score.
        """
        try:
            if self.sentiment_pipeline is None:
                return {"label": "neutral", "score": 0.5}

            # Get sentiment prediction
            sentiment_result = self.sentiment_pipeline(text)[0]

            # Return the sentiment label and score
            return {
                "label": sentiment_result["label"].lower(),
                "score": float(sentiment_result["score"]),
            }
        except Exception as e:
            print(f"WARNING [NLUInferencer]: Sentiment analysis failed: {e}")
            return {"label": "neutral", "score": 0.5}

    def _predict_entities(self, text):
        """
        Predict the entities in the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            list: A list of dictionaries, each containing an entity type and value.
        """
        try:
            # Tokenize the input
            word_tokens = text.split()
            inputs = self.entity_tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            # Get word_ids from tokenizer
            word_ids = inputs.word_ids(batch_index=0)

            # Predict
            with torch.no_grad():
                outputs = self.entity_model(**inputs)

            # Get predictions for each token
            logits = outputs.logits[0].cpu().numpy()
            predictions = np.argmax(logits, axis=1)

            # Map predicted IDs to tags
            tags = [self.id2tag.get(pred, "O") for pred in predictions]

            # Align predictions to words and collect entity groups
            word_predictions = []
            for i in range(len(word_ids)):
                # Skip special tokens (CLS, SEP, PAD)
                if word_ids[i] is None:
                    continue

                # If this is the first token of a word, add it to word_predictions
                if i == 0 or word_ids[i] != word_ids[i - 1]:
                    word_predictions.append((word_tokens[word_ids[i]], tags[i]))

            # Extract entities from BIO tags
            entities = []
            current_entity_tokens = []
            current_entity_type = None

            for word, tag in word_predictions:
                if tag.startswith("B-"):
                    # If we were processing a previous entity, add it to the result
                    if current_entity_type is not None:
                        entity_value = " ".join(current_entity_tokens)
                        entities.append(
                            {"entity": current_entity_type, "value": entity_value}
                        )

                    # Start a new entity
                    current_entity_tokens = [word]
                    current_entity_type = tag[2:]  # Remove the "B-" prefix

                elif tag.startswith("I-"):
                    # Only add to the current entity if its type matches
                    if (
                        current_entity_type is not None
                        and tag[2:] == current_entity_type
                    ):
                        current_entity_tokens.append(word)
                    # Otherwise, treat as O (misaligned I tag)

                elif tag == "O":
                    # If we were processing an entity, add it to the result
                    if current_entity_type is not None:
                        entity_value = " ".join(current_entity_tokens)
                        entities.append(
                            {"entity": current_entity_type, "value": entity_value}
                        )
                        current_entity_tokens = []
                        current_entity_type = None

            # Don't forget the last entity if we're still building one
            if current_entity_type is not None:
                entity_value = " ".join(current_entity_tokens)
                entities.append({"entity": current_entity_type, "value": entity_value})

            return entities

        except Exception as e:
            return []
