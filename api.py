from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, List, Optional, Any
from inference import NLUInferencer
from dialog_manager import DialogManager

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

class TowingState:
    def __init__(self):
        self.locations = {}  # conversation_id -> location
        self.confirmations = set()  # set of confirmed conversation_ids

# Initialize the NLU model
try:
    nlu_model = NLUInferencer()
    logger.info("NLU model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load NLU model: {e}")
    raise RuntimeError(f"Failed to load NLU model: {e}")

# Initialize the Dialog Manager
try:
    dialog_manager = DialogManager()
    logger.info("Dialog Manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Dialog Manager: {e}")
    dialog_manager = None  # Continue without dialog management

@app.get("/")
async def root():
    return {"message": "Welcome to the NLU Chatbot API"}

@app.post("/api/nlu", response_model=NLUResponse)
async def process_text(request: NLURequest):
    logger.info(f"Processing text: {request.text}")
    
    try:
        # Process the text through NLU
        result = nlu_model.predict(request.text)
        logger.info(f"NLU result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"NLU processing error: {str(e)}")

@app.post("/api/dialog", response_model=DialogResponse)
async def process_dialog(request: DialogRequest):
    logger.info(f"Processing dialog: {request.text}")
    
    # Create a global state for towing conversations if it doesn't exist
    if not hasattr(app, "towing_state"):
        app.towing_state = TowingState()
    
    # Hard-coded towing dialog flow
    text = request.text.lower()
    cid = request.conversation_id or "default"
    
    # Check for towing/breakdown related words
    towing_keywords = ["tow", "broke down", "broken down", "not starting", "won't start", "flat tire"]
    
    # If this conversation already has a location stored
    if cid in app.towing_state.locations:
        location = app.towing_state.locations[cid]
        
        # If they've already confirmed 
        if cid in app.towing_state.confirmations:
            return {
                "text": f"Your tow truck has been dispatched to {location} and should arrive within 30-45 minutes. Is there anything else you need help with?",
                "conversation_id": cid
            }
        
        # User is confirming
        if any(word in text for word in ["yes", "correct", "right", "confirm", "ok", "okay", "sure", "good"]):
            app.towing_state.confirmations.add(cid)
            return {
                "text": f"Great! I've dispatched a tow truck to {location}. It should arrive within 30-45 minutes. Is there anything else you need help with?",
                "conversation_id": cid
            }
        
        # User is declining or changing location
        if any(word in text for word in ["no", "wrong", "incorrect", "change", "different"]):
            app.towing_state.locations.pop(cid)
            return {
                "text": "I understand. Could you please provide the correct location where you need the tow truck?",
                "conversation_id": cid
            }
        
        # Otherwise, we're waiting for confirmation
        return {
            "text": f"Just to confirm, you need a tow truck at {location}. Is that correct?",
            "conversation_id": cid
        }
    
    # Initial towing request
    elif any(keyword in text for keyword in towing_keywords):
        # Check if location is in the message
        location_markers = ["at ", "on ", "near ", "by ", "close to "]
        location = ""
        
        for marker in location_markers:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) > 1:
                    location = marker + parts[1]
                    break
        
        if location:
            app.towing_state.locations[cid] = location
            return {
                "text": f"I see you're {location}. I'll send a tow truck to that location. Is that correct?",
                "conversation_id": cid
            }
        else:
            return {
                "text": "I can help you with towing. Can you please provide your current location so I can dispatch a tow truck?",
                "conversation_id": cid
            }
    
    # Location provided after towing request
    elif app.towing_state.locations.get(cid, "") == "":
        # Assume this is a location response
        app.towing_state.locations[cid] = request.text
        return {
            "text": f"Thank you. I'll send a tow truck to {request.text}. Is that correct?",
            "conversation_id": cid
        }
    
    # Fallback to the regular dialog manager if available
    if dialog_manager:
        try:
            # Process the text through the dialog manager
            result = dialog_manager.process_turn(request.text, request.conversation_id)
            logger.info(f"Dialog result: {result}")
            
            # In case of error or None result, provide a fallback response
            if result is None:
                return {
                    "text": "I'm sorry, I'm having trouble processing your request right now. Could you try again?",
                    "conversation_id": request.conversation_id or "default_session"
                }
            
            # Extract the response
            response_text = "I'm sorry, I didn't understand that."
            
            # Try different fields where the response might be
            if "bot_response" in result:
                response_text = result["bot_response"]
            elif "response" in result:
                response_text = result["response"]
            elif "action" in result and "text" in result["action"]:
                response_text = result["action"]["text"]
            
            # If we're supposed to ask for a slot, do simple default responses
            if "action" in result and result["action"].get("type") == "REQUEST_SLOT":
                slot = result["action"].get("slot_name", "")
                if slot == "pickup_location":
                    response_text = "Please provide your current location so I can send a tow truck."
                elif "destination" in slot:
                    response_text = "Where would you like your vehicle to be towed to?"
                elif "vehicle" in slot:
                    response_text = f"What is the {slot.replace('vehicle_', '').replace('_', ' ')} of your vehicle?"
            
            return {
                "text": response_text,
                "conversation_id": request.conversation_id or "default_session"
            }
        except Exception as e:
            logger.error(f"Error processing dialog: {e}")
            return {
                "text": "I apologize, but I encountered an error. Let's start fresh. How can I help you?",
                "conversation_id": request.conversation_id or "default_session"
            }
    else:
        return {
            "text": "I'm sorry, I'm having trouble processing your request right now. Could you try again later?",
            "conversation_id": request.conversation_id or "default_session"
        }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 