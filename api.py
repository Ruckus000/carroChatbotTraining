from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, List, Optional, Any, Set
from inference import NLUInferencer
from dialog_manager import DialogManager
import uuid
import os
import time
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nlu-api")

# Global connection tracking
active_clients: Dict[str, Dict[str, Any]] = {}
recent_clients: List[Dict[str, Any]] = []
MAX_RECENT_CLIENTS = 10  # How many recent clients to remember

# Initialize FastAPI app
app = FastAPI(
    title="NLU Chatbot API",
    description="API for Natural Language Understanding in a chatbot system",
    version="1.0.0",
)

# Client connection tracking middleware
class ClientTrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        client_id = f"{client_host}:{request.headers.get('user-agent', 'unknown')}"
        
        # Track the client connection
        timestamp = datetime.now().isoformat()
        endpoint = request.url.path
        
        # Detect platform
        platform = self._detect_platform(request.headers.get("user-agent", ""), request)
        
        # Add or update active client
        active_clients[client_id] = {
            "ip": client_host,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "last_seen": timestamp,
            "last_endpoint": endpoint,
            "platform": platform,
        }
        
        # Simple RN detection for logging
        if platform == "React Native":
            logger.info(f"REACT_NATIVE_CLIENT: {client_host} connected to {endpoint} - this is a mobile app connection")
        
        # Log the connection
        logger.info(f"API_CLIENT_CONNECTION: {client_host} | {endpoint} | {platform}")
        
        # Process the request
        response = await call_next(request)
        return response
    
    def _detect_platform(self, user_agent: str, request: Request) -> str:
        """Detect the platform from user agent string or headers"""
        user_agent = user_agent.lower()
        
        # Check X-Platform header first
        platform_header = request.headers.get("X-Platform", "").lower()
        if "react" in platform_header:
            return "React Native"
        
        if "react-native" in user_agent or "expo" in user_agent:
            return "React Native"
        elif "android" in user_agent:
            return "Android"
        elif "iphone" in user_agent or "ipad" in user_agent or "ipod" in user_agent:
            return "iOS"
        elif "mozilla" in user_agent:
            return "Browser"
        else:
            return "Unknown"

# Add middleware
app.add_middleware(ClientTrackingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define request and response models
class NLURequest(BaseModel):
    text: str


class EntityModel(BaseModel):
    entity: str
    value: str


class IntentModel(BaseModel):
    name: str
    confidence: float


class NLUResponse(BaseModel):
    text: str
    intent: IntentModel
    entities: List[EntityModel]


class DialogRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None


class DialogResponse(BaseModel):
    text: str
    conversation_id: str


# Initialize the NLU model
try:
    nlu_inferencer = NLUInferencer()
    logger.info("NLU model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load NLU model: {e}")
    raise RuntimeError(f"Failed to load NLU model: {e}")

# Initialize the Dialog Manager with NLU
try:
    dialog_manager = DialogManager(nlu_inferencer=nlu_inferencer)
    logger.info("Dialog Manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Dialog Manager: {e}")
    raise RuntimeError(f"Failed to initialize Dialog Manager: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to the NLU Chatbot API"}


@app.post("/api/nlu", response_model=NLUResponse)
async def process_text(request: NLURequest):
    logger.info(f"API_REQUEST: /api/nlu | Text: '{request.text}'")

    try:
        # Process the text through NLU
        result = nlu_inferencer.predict(request.text)
        logger.info(f"API_RESPONSE: /api/nlu | Intent: '{result.intent.name}', Confidence: {result.intent.confidence}")
        return result
    except Exception as e:
        logger.error(f"API_ERROR: /api/nlu | Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"NLU processing error: {str(e)}")


@app.post("/api/dialog", response_model=DialogResponse)
async def process_dialog(request: DialogRequest):
    """Process a dialog turn through the dialog manager."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    logger.info(f"API_REQUEST: /api/dialog | ConvID: {conversation_id} | User: '{request.text}'")

    try:
        # Process the turn through the dialog manager
        result = dialog_manager.process_turn(request.text, conversation_id)

        # Extract the response text
        if isinstance(result, dict) and "bot_response" in result:
            response_text = result.get("bot_response", "Error: No response generated.")
            # Optional: Log state or action details for debugging
            logger.debug(f"DM Action for {conversation_id}: {result.get('action')}")
        else:
            logger.error(
                f"DialogManager returned unexpected/invalid result for {conversation_id}: {result}"
            )
            response_text = "I'm sorry, I encountered an internal processing issue. Could you please try again?"

        # Ensure response_text is always a string
        if not isinstance(response_text, str):
            logger.error(
                f"Generated response is not a string for {conversation_id}: {type(response_text)}"
            )
            response_text = "Error: Invalid response format."

        logger.info(f"API_RESPONSE: /api/dialog | ConvID: {conversation_id} | Bot: '{response_text}'")
        return DialogResponse(text=response_text, conversation_id=conversation_id)

    except Exception as e:
        # Log the full exception details for server-side debugging with special prefix
        logger.error(
            f"API_ERROR: /api/dialog | ConvID: {conversation_id} | Error processing: {e}",
            exc_info=True,
        )
        # Return a generic, user-friendly error response
        return DialogResponse(
            text="I apologize, but an unexpected internal error occurred. Please try again later.",
            conversation_id=conversation_id,
        )


@app.post("/chat", include_in_schema=False)
async def legacy_chat_endpoint(request: dict, req: Request = None):
    """Legacy endpoint that redirects to the new format."""
    logger.info(f"API_REQUEST: /chat | Legacy chat endpoint | Data: {request}")
    
    # Check headers for React Native platform
    is_react_native = False
    client_ip = "unknown"
    
    if req and req.headers:
        user_agent = req.headers.get("user-agent", "")
        platform_header = req.headers.get("x-platform", "")
        
        if "react" in user_agent.lower() or "react" in platform_header.lower():
            is_react_native = True
           
        if req.client:
            client_ip = req.client.host
    
    # Create a proper request for the /api/dialog endpoint
    dialog_request = DialogRequest(
        text=request.get("text", ""),
        conversation_id=request.get("conversationId", None)
    )
    
    try:
        # Process using the dialog endpoint
        response = await process_dialog(dialog_request)
        
        # Choose prefix based on client type
        if is_react_native or (client_ip != "127.0.0.1" and client_ip != "unknown"):
            prefix = "\n🔵🔵🔵 REACT NATIVE APP 🔵🔵🔵"
        else:
            prefix = "\n=== CHAT MESSAGE ===="
        
        # Print terminal message for all chat requests - simple logging
        print(f"{prefix}\nUSER: {request.get('text', '')}\nBOT: {response.text}\n{'=' * 30}\n")
        
        # Return the legacy format response
        result = {
            "response": response.text,
            "conversationId": response.conversation_id
        }
        
        logger.info(f"API_RESPONSE: /chat | Legacy format | ConvID: {response.conversation_id}")
        return result
    except Exception as e:
        logger.error(f"Error in legacy chat endpoint: {e}")
        return {"error": str(e)}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/health")
async def legacy_health_check():
    """Legacy health endpoint that returns the format expected by existing consumers."""
    return {
        "status": "ok", 
        "message": "Carro Backend API health check", 
        "components": {
            "config": "ok", 
            "chat_service": "ok"
        }
    }


@app.get("/api/connections", include_in_schema=False)
async def get_connections():
    """Get a list of current connections."""
    # Filter to active clients only (within the last minute)
    now = datetime.now()
    active = {}
    for client_id, client in active_clients.items():
        last_seen_dt = datetime.fromisoformat(client["last_seen"])
        if (now - last_seen_dt).total_seconds() < 60:
            active[client_id] = {
                "ip": client["ip"],
                "platform": client["platform"],
                "last_endpoint": client["last_endpoint"],
                "last_seen": client["last_seen"],
            }
    
    return {"active": active}


if __name__ == "__main__":
    # Read port from environment variable, default to 8001 if not set
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0") # Optional: make host configurable too
    print(f"INFO:     Starting NLU API on {host}:{port}") # Add this for clarity
    uvicorn.run("api:app", host=host, port=port, reload=False)
