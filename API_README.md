# NLU Chatbot Integration Guide

This document outlines how to integrate the NLU chatbot with your React Native frontend. The NLU API server is already running via Docker and provides intent detection and entity extraction for text messages.

## API Overview

The NLU chatbot API is a standalone service that:

- Analyzes text to determine user intent
- Extracts entities from user messages
- Provides a simple REST interface

## API Server Status

The NLU chatbot API server is **already running via Docker** at `http://localhost:8000`. You don't need to set up or deploy this service - it's ready to use.

## API Endpoints

The following endpoints are available:

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy"}`
- **Purpose**: Verify the API server is running and responsive

### Process Text

- **URL**: `/api/nlu`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
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
- **Error Response**:
  ```json
  {
    "detail": "NLU processing error: [error message]"
  }
  ```

## Testing the API

You can test the already-running API using cURL:

```bash
# Health check
curl -X GET http://localhost:8000/api/health

# Process text
curl -X POST http://localhost:8000/api/nlu \
  -H "Content-Type: application/json" \
  -d '{"text": "My car broke down on the highway, I need a tow"}'
```

## React Native Integration

### 1. Create API Service

Create a service to handle API calls in your React Native project:

```javascript
// src/services/nluService.js
const API_URL = 'http://localhost:8000/api/nlu'

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
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Network response was not ok')
    }

    return await response.json()
  } catch (error) {
    console.error('Error processing message:', error)
    return {
      text: text,
      intent: { name: 'error', confidence: 1.0 },
      entities: [],
    }
  }
}

// Check if API is available
export const checkApiHealth = async () => {
  try {
    const response = await fetch(
      `${API_URL.substring(0, API_URL.lastIndexOf('/'))}/health`,
      {
        method: 'GET',
      }
    )
    return response.ok
  } catch (error) {
    console.error('API health check failed:', error)
    return false
  }
}
```

### 2. Chat Component Implementation

Use the service in your chat component:

```javascript
// src/components/Chat.js
import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  ActivityIndicator,
} from 'react-native'
import { processUserMessage, checkApiHealth } from '../services/nluService'

const Chat = () => {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiAvailable, setApiAvailable] = useState(true)

  // Check API health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      const isHealthy = await checkApiHealth()
      setApiAvailable(isHealthy)
    }
    checkHealth()
  }, [])

  const handleSend = async () => {
    if (!inputText.trim() || !apiAvailable) return

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
    }

    setMessages((prevMessages) => [...prevMessages, userMessage])
    setInputText('')
    setIsLoading(true)

    try {
      // Process with NLU API
      const nluResult = await processUserMessage(inputText)

      // Create response based on intent and entities
      let responseText = ''

      if (nluResult.intent.name.includes('fallback')) {
        responseText = "I'm not sure I understand. Could you rephrase that?"
      } else {
        // Here you would implement logic to generate responses based on intents
        responseText = `I detected: ${nluResult.intent.name}`

        // Example of using entities in response
        if (nluResult.entities.length > 0) {
          const entityText = nluResult.entities
            .map((entity) => `${entity.entity}: ${entity.value}`)
            .join(', ')
          responseText += `\nWith entities: ${entityText}`
        }
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: responseText,
        sender: 'bot',
        nluData: nluResult,
      }

      setMessages((prevMessages) => [...prevMessages, botMessage])
    } catch (error) {
      // Handle error
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, I'm having trouble connecting right now.",
        sender: 'bot',
        isError: true,
      }
      setMessages((prevMessages) => [...prevMessages, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <View style={styles.container}>
      {!apiAvailable && (
        <View style={styles.apiWarning}>
          <Text style={styles.apiWarningText}>
            Cannot connect to the chatbot API. Please check your connection.
          </Text>
        </View>
      )}

      <FlatList
        data={messages}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View
            style={[
              styles.messageBubble,
              item.sender === 'user' ? styles.userBubble : styles.botBubble,
              item.isError && styles.errorBubble,
            ]}
          >
            <Text style={styles.messageText}>{item.text}</Text>
          </View>
        )}
        style={styles.messageList}
      />

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type a message..."
          placeholderTextColor="#999"
          onSubmitEditing={handleSend}
          editable={!isLoading && apiAvailable}
        />
        {isLoading ? (
          <ActivityIndicator
            size="small"
            color="#0084ff"
            style={styles.sendButton}
          />
        ) : (
          <TouchableOpacity
            style={[styles.sendButton, !apiAvailable && styles.disabledButton]}
            onPress={handleSend}
            disabled={!apiAvailable}
          >
            <Text style={styles.sendButtonText}>Send</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  apiWarning: {
    backgroundColor: '#ffcccc',
    padding: 10,
    alignItems: 'center',
  },
  apiWarningText: {
    color: '#cc0000',
    fontSize: 14,
  },
  messageList: {
    flex: 1,
    padding: 10,
  },
  messageBubble: {
    borderRadius: 20,
    padding: 15,
    marginVertical: 5,
    maxWidth: '80%',
    alignSelf: 'flex-start',
  },
  userBubble: {
    backgroundColor: '#0084ff',
    alignSelf: 'flex-end',
  },
  botBubble: {
    backgroundColor: '#e5e5e5',
  },
  errorBubble: {
    backgroundColor: '#ffcccc',
  },
  messageText: {
    fontSize: 16,
    color: '#000',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#ddd',
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginRight: 10,
    fontSize: 16,
  },
  sendButton: {
    backgroundColor: '#0084ff',
    borderRadius: 20,
    width: 60,
    justifyContent: 'center',
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#cccccc',
  },
  sendButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
})

export default Chat
```

### 3. Environment Configuration

Configure API URLs for different environments:

```javascript
// src/config/api.js
const API_CONFIG = {
  development: {
    // For iOS simulator
    ios: 'http://localhost:8000/api',
    // For Android emulator (redirects to host machine's localhost)
    android: 'http://10.0.2.2:8000/api',
  },
  production: {
    // Your production API URL
    ios: 'https://your-production-api.com/api',
    android: 'https://your-production-api.com/api',
  },
}

// Determine platform and environment
const platform = Platform.OS
const environment = __DEV__ ? 'development' : 'production'

// Export the appropriate URL
export const API_BASE_URL = API_CONFIG[environment][platform]
export const API_NLU_URL = `${API_BASE_URL}/nlu`
export const API_HEALTH_URL = `${API_BASE_URL}/health`
```

Then update your service:

```javascript
// src/services/nluService.js
import { API_NLU_URL, API_HEALTH_URL } from '../config/api';

export const processUserMessage = async (text) => {
  try {
    const response = await fetch(API_NLU_URL, {
      // ... rest of the code
    });
    // ...
  }
};

export const checkApiHealth = async () => {
  try {
    const response = await fetch(API_HEALTH_URL);
    // ...
  }
};
```

## Testing in Development

When testing locally:

1. **Ensure the Docker container is running**:
   You should see the Docker container logs indicating the API is available on port 8000

2. **For iOS simulator**: Use 'http://localhost:8000/api' as the base URL
3. **For Android emulator**: Use 'http://10.0.2.2:8000/api' as the base URL
4. **For physical devices on the same WiFi**: Use your computer's local IP address: 'http://192.168.x.x:8000/api'

## Important Notes

1. **The API is already running** - You don't need to start it manually, the Docker container is handling this
2. **API Response Format** - The API returns both intent and entities, use them to build your response logic
3. **Error Handling** - Always implement proper error handling as shown in the example
4. **Connectivity Issues** - Test your app's behavior when the API is unreachable
