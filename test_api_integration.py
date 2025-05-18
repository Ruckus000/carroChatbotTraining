import pytest
import httpx # Async HTTP client
import asyncio
import uuid
import os
from typing import Optional, Dict, Any, List # Ensure typing.Optional is imported
from pydantic import BaseModel, ValidationError, Field

# Define Pydantic models for expected API response structures
class SentimentModel(BaseModel):
    label: str
    score: float
    
    def validate_label(self) -> bool:
        """Validate that the label is one of the expected values (case-insensitive)"""
        normalized_label = self.label.upper()
        return normalized_label in ["POSITIVE", "NEGATIVE", "NEUTRAL", "POS", "NEG", "NEU"]

class IntentModel(BaseModel):
    name: str
    confidence: float

class NLUResponseModel(BaseModel):
    text: str
    intent: IntentModel
    entities: List[Dict[str, Any]]
    sentiment: Optional[SentimentModel]

# Determine API port from environment, default to 8001 (matching api.py)
API_PORT = os.environ.get("PORT", "8001")
API_BASE_URL = f"http://127.0.0.1:{API_PORT}"

NLU_ENDPOINT = f"{API_BASE_URL}/api/nlu"
DIALOG_ENDPOINT = f"{API_BASE_URL}/api/dialog"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"
LEGACY_CHAT_ENDPOINT = f"{API_BASE_URL}/chat" # For legacy compatibility test
LEGACY_HEALTH_ENDPOINT = f"{API_BASE_URL}/health" # For legacy compatibility test

print(f"INFO [TestAPI]: Running API integration tests against {API_BASE_URL}")

# Helper function to send requests
async def _send_request(client: httpx.AsyncClient, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> httpx.Response:
    headers = {"Content-Type": "application/json", "X-Platform": "pytest-integration-test"}
    try:
        if method.upper() == 'GET':
            response = await client.get(url, headers=headers, timeout=20) # Increased timeout
        elif method.upper() == 'POST':
            response = await client.post(url, json=payload, headers=headers, timeout=20)
        else:
            pytest.fail(f"Unsupported HTTP method: {method}")

        print(f"DEBUG [TestAPI]: {method} {url} | Payload: {payload} | Status: {response.status_code} | Response: {response.text[:300]}...")
        response.raise_for_status()
        return response
    except httpx.RequestError as exc:
        print(f"ERROR [TestAPI]: Request failed for {method} {url}: {exc}")
        pytest.fail(f"HTTP Request failed: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"ERROR [TestAPI]: HTTP Error for {method} {url}: {exc.response.status_code} - {exc.response.text}")
        pytest.fail(f"HTTP Error {exc.response.status_code} - {exc.response.text}")

# Schema validation helper
def validate_schema(data: Dict[str, Any], required_fields: List[str], field_types: Dict[str, type] = None):
    """Validate that a dictionary contains all required fields and optional type checks."""
    for field in required_fields:
        assert field in data, f"Required field '{field}' is missing"
    
    if field_types:
        for field, expected_type in field_types.items():
            if field in data and data[field] is not None:
                assert isinstance(data[field], expected_type), f"Field '{field}' should be of type {expected_type.__name__}, got {type(data[field]).__name__}"

# Advanced validation with Pydantic models
def validate_nlu_response_model(data: Dict[str, Any]) -> None:
    """Validate NLU response using Pydantic models for robust schema validation."""
    try:
        nlu_response = NLUResponseModel(**data)
        
        # Additional validation on the sentiment label if present
        if nlu_response.sentiment:
            assert nlu_response.sentiment.validate_label(), f"Invalid sentiment label: {nlu_response.sentiment.label}"
            
        return nlu_response
    except ValidationError as e:
        pytest.fail(f"NLU response failed Pydantic validation: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during NLU response validation: {e}")

@pytest.mark.asyncio
async def test_api_health_check():
    """Test the primary /api/health endpoint."""
    async with httpx.AsyncClient() as client:
        response = await _send_request(client, 'GET', HEALTH_ENDPOINT)
        assert response.status_code == 200
        data = response.json()
        validate_schema(data, ["status"])
        assert data["status"] in ["healthy", "ok"], f"Expected health status to be 'healthy' or 'ok', got '{data['status']}'"
    print("SUCCESS [TestAPI]: /api/health check passed.")

@pytest.mark.asyncio
async def test_legacy_health_check():
    """Test the legacy /health endpoint."""
    async with httpx.AsyncClient() as client:
        response = await _send_request(client, 'GET', LEGACY_HEALTH_ENDPOINT)
        assert response.status_code == 200
        data = response.json()
        validate_schema(data, ["status", "components"])
        assert data["status"] in ["ok", "healthy"], f"Expected health status to be 'ok' or 'healthy', got '{data['status']}'"
        assert isinstance(data["components"], dict), "Expected 'components' to be a dictionary"
    print("SUCCESS [TestAPI]: /health legacy check passed.")

@pytest.mark.asyncio
async def test_api_nlu_endpoint_basic():
    """Test /api/nlu with a simple input."""
    async with httpx.AsyncClient() as client:
        payload = {"text": "I need a tow truck for my Honda."}
        response = await _send_request(client, 'POST', NLU_ENDPOINT, payload)
        data = response.json()
        
        # Use Pydantic for schema validation
        nlu_response = validate_nlu_response_model(data)
        
        # Additional content validation
        assert nlu_response.text == payload["text"], "Input text should be echoed in response"
        assert isinstance(nlu_response.entities, list), "Entities should be a list"
        assert nlu_response.sentiment is not None, "Sentiment should not be None"
        
    print("SUCCESS [TestAPI]: /api/nlu basic test passed.")

@pytest.mark.asyncio
async def test_api_nlu_sentiment_positive():
    """Test /api/nlu for positive sentiment."""
    async with httpx.AsyncClient() as client:
        payload = {"text": "This is a wonderful and fantastic service!"}
        response = await _send_request(client, 'POST', NLU_ENDPOINT, payload)
        data = response.json()
        
        # Use Pydantic for validation
        nlu_response = validate_nlu_response_model(data)
        
        # Check sentiment is positive using case-insensitive comparison
        sentiment_label = nlu_response.sentiment.label.upper()
        assert sentiment_label in ["POSITIVE", "POS"], f"Expected positive sentiment, got '{sentiment_label}'"
    print("SUCCESS [TestAPI]: /api/nlu positive sentiment passed.")

@pytest.mark.asyncio
async def test_api_nlu_sentiment_negative():
    """Test /api/nlu for negative sentiment."""
    async with httpx.AsyncClient() as client:
        payload = {"text": "I am very angry and frustrated with this problem."}
        response = await _send_request(client, 'POST', NLU_ENDPOINT, payload)
        data = response.json()
        
        # Use Pydantic for validation
        nlu_response = validate_nlu_response_model(data)
        
        # Check sentiment is negative using case-insensitive comparison
        sentiment_label = nlu_response.sentiment.label.upper()
        assert sentiment_label in ["NEGATIVE", "NEG"], f"Expected negative sentiment, got '{sentiment_label}'"
    print("SUCCESS [TestAPI]: /api/nlu negative sentiment passed.")

@pytest.mark.asyncio
async def test_api_dialog_full_towing_flow():
    """Test a complete towing conversation flow via /api/dialog."""
    async with httpx.AsyncClient() as client:
        conv_id = f"test_towing_flow_{uuid.uuid4().hex[:8]}"

        # Turn 1: Initial request - Check if asks for location
        response = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "My car is broken, I need a tow.", "conversation_id": conv_id})
        data = response.json()
        validate_schema(data, ["conversation_id", "text"])
        assert data["conversation_id"] == conv_id
        assert any(keyword in data["text"].lower() for keyword in ["location", "where", "address", "place", "pickup"]), "Response should ask for location"

        # Turn 2: Provide pickup location - Check if asks for destination
        response = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "I am at 789 Rescue Road.", "conversation_id": conv_id})
        data = response.json()
        assert any(keyword in data["text"].lower() for keyword in ["destination", "where to", "take it", "drop off"]), "Response should ask for destination"

        # Turn 3: Provide destination - Check if asks for vehicle details
        response = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "Take it to Reliable Repairs LLC.", "conversation_id": conv_id})
        data = response.json()
        assert any(keyword in data["text"].lower() for keyword in ["vehicle", "car", "make", "model"]), "Response should ask for vehicle information"

        # Turn 4: Provide vehicle details - Check if asks for confirmation
        response = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "It's a 2021 Toyota Rav4.", "conversation_id": conv_id})
        data = response.json()
        assert any(keyword in data["text"].lower() for keyword in ["confirm", "verification", "correct", "right"]), "Response should ask for confirmation"
        
        # Verify information is reflected back
        assert "2021" in data["text"] or "toyota" in data["text"].lower() or "rav4" in data["text"].lower(), "Response should mention the vehicle"
        assert "rescue road" in data["text"].lower() or "789" in data["text"], "Response should mention the pickup location"
        assert "reliable repairs" in data["text"].lower() or "llc" in data["text"].lower(), "Response should mention the destination"

        # Turn 5: Confirm - Check if confirms booking
        response = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "Yes, that is correct, please book it.", "conversation_id": conv_id})
        data = response.json()
        assert any(phrase in data["text"].lower() for phrase in ["confirmed", "booked", "dispatched", "all set", "on the way", "thank you"]), "Response should confirm the booking"
    print("SUCCESS [TestAPI]: /api/dialog towing flow test passed.")

@pytest.mark.asyncio
async def test_legacy_chat_endpoint_flow():
    """Test the legacy /chat endpoint."""
    async with httpx.AsyncClient() as client:
        conv_id = f"legacy_chat_test_{uuid.uuid4().hex[:8]}"
        # Turn 1
        response1 = await _send_request(client, 'POST', LEGACY_CHAT_ENDPOINT,
                                        {"text": "Help, I need a jump start", "conversationId": conv_id})
        data1 = response1.json()
        validate_schema(data1, ["conversationId", "response"])
        assert data1["conversationId"] == conv_id
        assert any(keyword in data1["response"].lower() for keyword in ["location", "where", "address", "place"]), "Response should ask for location"

        # Turn 2
        response2 = await _send_request(client, 'POST', LEGACY_CHAT_ENDPOINT,
                                        {"text": "At the library on Main St", "conversationId": conv_id})
        data2 = response2.json()
        # Check that conversation is progressing to either vehicle details or confirmation
        assert any(keyword in data2["response"].lower() for keyword in ["vehicle", "car", "make", "model", "confirm", "verification", "correct"]), "Response should ask for vehicle details or confirmation"
    print("SUCCESS [TestAPI]: /chat legacy endpoint flow test passed.")

@pytest.mark.asyncio
async def test_dialog_state_persistence():
    """Test that different conversation IDs maintain separate states."""
    async with httpx.AsyncClient() as client:
        conv_id_A = f"state_A_{uuid.uuid4().hex[:8]}"
        conv_id_B = f"state_B_{uuid.uuid4().hex[:8]}"

        # Conv A - Turn 1 (Towing)
        res_A1 = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "I need a tow", "conversation_id": conv_id_A})
        data_A1 = res_A1.json()
        validate_schema(data_A1, ["conversation_id", "text"])
        assert any(keyword in data_A1["text"].lower() for keyword in ["location", "where", "address", "pickup"]), "Response should ask for location (towing flow)"

        # Conv B - Turn 1 (Appointment)
        res_B1 = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "Book an oil change", "conversation_id": conv_id_B})
        data_B1 = res_B1.json()
        validate_schema(data_B1, ["conversation_id", "text"])
        # For appointment: might ask for location, vehicle, or date
        appointment_keywords = ["location", "where", "vehicle", "car", "date", "when", "time"]
        assert any(keyword in data_B1["text"].lower() for keyword in appointment_keywords), "Response should ask for appointment details"

        # Conv A - Turn 2 (Towing)
        res_A2 = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "My location is 111 Emergency Ave", "conversation_id": conv_id_A})
        data_A2 = res_A2.json()
        assert any(keyword in data_A2["text"].lower() for keyword in ["destination", "where to", "take it", "drop off"]), "Response should ask for destination (towing flow)"

        # Conv B - Turn 2 (Appointment)
        res_B2 = await _send_request(client, 'POST', DIALOG_ENDPOINT, {"text": "Downtown Service Center please", "conversation_id": conv_id_B})
        data_B2 = res_B2.json()
        # For appointment: might ask for vehicle, date, or time (depending on flow)
        assert any(keyword in data_B2["text"].lower() for keyword in ["vehicle", "car", "date", "when", "time", "day"]), "Response should ask for appointment details"
        
        # Verify states are distinct (responses should be different for different flows)
        assert data_A2["text"] != data_B2["text"], "Different conversation flows should yield different responses"
    print("SUCCESS [TestAPI]: Dialog state persistence passed.") 