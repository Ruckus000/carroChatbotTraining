# NLU Chatbot API

This API exposes the NLU (Natural Language Understanding) functionality of the chatbot system through REST endpoints. It provides intent detection and entity extraction for incoming text messages.

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements-api.txt
   ```

2. **Start the API Server**:

   ```bash
   python api.py
   ```

   The server will start on `http://localhost:8000`.

3. **Access API Documentation**:
   Once the server is running, you can access the interactive API documentation at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy"}`

### Process Text

- **URL**: `/api/nlu`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "text": "Your text message here"
  }
  ```
- **Response**:
  ```json
  {
    "text": "Your text message here",
    "intent": {
      "name": "detected_intent_name",
      "confidence": 0.9876
    },
    "entities": [
      {
        "entity": "entity_type",
        "value": "entity_value"
      }
    ]
  }
  ```

## Testing the API

A test script is provided to quickly check if the API is working correctly:

```bash
# Run with default test message
python test_api.py

# Run with custom message
python test_api.py "I need a tow for my Tesla Model 3"
```

## Integrating with React Native

To integrate this API with your React Native application, here's a simple example:

```javascript
// chatService.js
const API_URL = 'http://your-server-address:8000/api/nlu'

export const processUserMessage = async (text) => {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    })

    if (!response.ok) {
      throw new Error('Network response was not ok')
    }

    return await response.json()
  } catch (error) {
    console.error('Error processing message:', error)
    return {
      intent: { name: 'error', confidence: 1.0 },
      entities: [],
    }
  }
}
```

## Production Deployment

For production deployment, consider:

1. **Using a WSGI server**:

   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
   ```

2. **Setting up HTTPS** for secure communication

3. **Configuring rate limiting** to prevent abuse

4. **Implementing authentication** for secure API access
