# NLU Chatbot Integration Guide

This document outlines how to integrate the NLU chatbot with your React Native frontend. The API server provides intent detection, entity extraction, and dialog management for conversations.

## API Overview

The chatbot API is a standalone service that:

- Analyzes text to determine user intent
- Extracts entities from user messages
- Manages conversation state across multiple turns
- Generates appropriate responses based on context
- Provides a simple REST interface

## API Server Status

The API server is **already running via Docker** at `http://localhost:8000`. You don't need to set up or deploy this service - it's ready to use.

## API Endpoints

The following endpoints are available:

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy"}`
- **Purpose**: Verify the API server is running and responsive

### Process Text (NLU Only)

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

### Process Dialog (Full Conversation)

- **URL**: `/api/dialog`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Body**:
  ```json
  {
    "text": "Your text message here",
    "conversation_id": "optional_conversation_id"
  }
  ```
- **Response**:
  ```json
  {
    "text": "Bot response text",
    "conversation_id": "conversation_id"
  }
  ```
- **Notes**:
  - If `conversation_id` is not provided, a new one will be generated
  - The same `conversation_id` should be used for multiple turns in the same conversation
  - The dialog endpoint maintains state between turns

## Testing the API

You can test the already-running API using cURL:

```bash
# Health check
curl -X GET http://localhost:8000/api/health

# Process text (NLU only)
curl -X POST http://localhost:8000/api/nlu \
  -H "Content-Type: application/json" \
  -d '{"text": "My car broke down on the highway, I need a tow"}'

# Process dialog (with conversation state)
curl -X POST http://localhost:8000/api/dialog \
  -H "Content-Type: application/json" \
  -d '{"text": "My car broke down on the highway, I need a tow", "conversation_id": "test123"}'
```

## React Native Integration

### 1. Create API Service

Create a service to handle API calls in your React Native project:

```javascript
// src/services/chatbotService.js
const API_BASE_URL = 'http://localhost:8000/api'
const API_NLU_URL = `${API_BASE_URL}/nlu`
const API_DIALOG_URL = `${API_BASE_URL}/dialog`
const API_HEALTH_URL = `${API_BASE_URL}/health`

// Process text through NLU only (no conversation state)
export const processTextNLU = async (text) => {
  try {
    const response = await fetch(API_NLU_URL, {
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

// Process dialog with conversation state
export const processDialog = async (text, conversationId = null) => {
  try {
    const body = { text }
    if (conversationId) {
      body.conversation_id = conversationId
    }

    const response = await fetch(API_DIALOG_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Network response was not ok')
    }

    return await response.json()
  } catch (error) {
    console.error('Error processing dialog:', error)
    return {
      text: 'Sorry, I encountered an error processing your request.',
      conversation_id: conversationId || 'error',
    }
  }
}

// Check if API is available
export const checkApiHealth = async () => {
  try {
    const response = await fetch(API_HEALTH_URL, {
      method: 'GET',
    })
    return response.ok
  } catch (error) {
    console.error('API health check failed:', error)
    return false
  }
}
```

### 2. Chat Component Implementation

Here's a basic chat component that uses the dialog endpoint to maintain conversation state:

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
import { processDialog, checkApiHealth } from '../services/chatbotService'

const Chat = () => {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiAvailable, setApiAvailable] = useState(true)
  const [conversationId, setConversationId] = useState(null)

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
      // Process with dialog API
      const result = await processDialog(inputText, conversationId)

      // Store conversation ID for subsequent messages
      if (result.conversation_id && !conversationId) {
        setConversationId(result.conversation_id)
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: result.text,
        sender: 'bot',
        conversationId: result.conversation_id,
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

Configure API URLs for different environments as in previous examples.

## Testing in Development

When testing locally:

1. **Ensure the Docker container is running**:
   You should see the Docker container logs indicating the API is available on port 8000

2. **For iOS simulator**: Use 'http://localhost:8000/api' as the base URL
3. **For Android emulator**: Use 'http://10.0.2.2:8000/api' as the base URL
4. **For physical devices on the same WiFi**: Use your computer's local IP address: 'http://192.168.x.x:8000/api'

## Important Notes

1. **The API is already running** - You don't need to start it manually, the Docker container is handling this
2. **Conversation State** - The dialog endpoint maintains state between turns when you provide the same conversation_id
3. **Dialog Flow** - The system will guide users through required information for different flows (towing, roadside assistance, etc.)
4. **Error Handling** - Always implement proper error handling as shown in the example
5. **Connectivity Issues** - Test your app's behavior when the API is unreachable
