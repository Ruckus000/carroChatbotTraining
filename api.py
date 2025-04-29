from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, List, Optional, Any
from inference import NLUInferencer

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

# Initialize the NLU model
try:
    nlu_model = NLUInferencer()
    logger.info("NLU model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load NLU model: {e}")
    raise RuntimeError(f"Failed to load NLU model: {e}")

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

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 