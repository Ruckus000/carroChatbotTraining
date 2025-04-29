from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, List, Optional, Any
from inference import NLUInferencer
from dialog_manager import DialogManager
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nlu-api")

# Initialize FastAPI app
app = FastAPI(
    title="NLU Chatbot API",
    description="API for Natural Language Understanding in a chatbot system",
    version="1.0.0",
)

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
    logger.info(f"Processing text: {request.text}")
    
    try:
        # Process the text through NLU
        result = nlu_inferencer.predict(request.text)
        logger.info(f"NLU result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"NLU processing error: {str(e)}")

@app.post("/api/dialog", response_model=DialogResponse)
async def process_dialog(request: DialogRequest):
    """Process a dialog turn through the dialog manager."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    logger.info(f"Processing dialog for conv_id '{conversation_id}': '{request.text}'")

    try:
        # Process the turn through the dialog manager
        result = dialog_manager.process_turn(request.text, conversation_id)
        
        # Extract the response text
        if isinstance(result, dict) and "bot_response" in result:
            response_text = result.get("bot_response", "Error: No response generated.")
            # Optional: Log state or action details for debugging
            logger.debug(f"DM Action for {conversation_id}: {result.get('action')}")
        else:
            logger.error(f"DialogManager returned unexpected/invalid result for {conversation_id}: {result}")
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
        return DialogResponse(
            text="I apologize, but an unexpected internal error occurred. Please try again later.",
            conversation_id=conversation_id
        )

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 