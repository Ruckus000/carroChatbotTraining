# NLU Chatbot Integration Guide

This document outlines how to integrate the NLU chatbot with your React Native frontend. The API server provides intent detection, entity extraction, sentiment analysis, and dialog management for conversations.

## IMPORTANT: Use Local API for Development

**CRITICAL UPDATE**: While there is a Docker container running on port 8000, it contains an older version of the API code. For development and testing, use the local API on port 8001 by running the provided script:

```bash
# Start the local API server on port 8001 (RECOMMENDED METHOD)
./scripts/start_api.sh
```

This script automatically:

- Sets up the Python virtual environment
- Installs required dependencies
- Checks for the trained model
- Stops any existing API server instances
- Starts the API server on port 8001 with proper logging

All examples and code in this document have been updated to use port 8001. If you need to use the Docker version on port 8000 for any reason, replace 8001 with 8000 in the examples.

## API Overview

The chatbot API is a standalone service that:

- Analyzes text to determine user intent
- Extracts entities from user messages
- Analyzes sentiment (emotional tone) of messages
- Manages conversation state across multiple turns
- Generates appropriate responses based on context
- Provides a simple REST interface

## API Server Status

~~The API server is **already running via Docker** at `http://localhost:8000`. You don't need to set up or deploy this service - it's ready to use.~~

**Updated**: Run the local API server using the `scripts/start_api.sh` script. This ensures you're using the latest version with all improvements and proper logging.

### Monitoring API Activity

When running with the `scripts/start_api.sh` script, all chat interactions will be displayed in the terminal with the following format:

```
🔵🔵🔵 REACT NATIVE APP 🔵🔵🔵
USER: [user message]
BOT: [bot response]
==============================
```

To ensure that React Native app connections are properly logged:

1. Set the `X-Platform: React Native` header in all API requests
2. Check the terminal where the API server is running to see the chat messages
3. Verify that connections from your app show the blue emoji indicators

If you don't see the messages in the terminal, ensure that:

1. You started the server with `./scripts/start_api.sh` (not directly with Python)
2. Your React Native app is sending the correct headers
3. You're looking at the correct terminal window where the server is running

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
    ],
    "sentiment": {
      "label": "negative",
      "score": 0.9876
    }
  }
  ```
- **Notes on Sentiment Analysis**:
  - The `sentiment` field is always included in the response
  - `label` will be one of: "positive", "negative", or "neutral"
  - `score` is a confidence value between 0 and 1
  - Higher scores indicate stronger sentiment confidence
  - The dialog system uses this information to adapt responses and handling of urgent situations
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
  - Sentiment analysis is performed internally and influences the response generation and dialog flow
  - For high negative sentiment in urgent situations (like towing requests), the dialog system will adapt to provide more empathetic responses and expedite the information collection process

## Testing the API

You can test the local API using cURL:

```bash
# Health check
curl -X GET http://localhost:8001/api/health \
  -H "X-Platform: React Native"

# Process text (NLU only)
curl -X POST http://localhost:8001/api/nlu \
  -H "Content-Type: application/json" \
  -H "X-Platform: React Native" \
  -d '{"text": "My car broke down on the highway, I need a tow"}'

# Process text with emotional content to see sentiment analysis
curl -X POST http://localhost:8001/api/nlu \
  -H "Content-Type: application/json" \
  -H "X-Platform: React Native" \
  -d '{"text": "I am so frustrated! My car broke down and I am stuck in the middle of nowhere!"}'

# Process dialog (with conversation state)
curl -X POST http://localhost:8001/api/dialog \
  -H "Content-Type: application/json" \
  -H "X-Platform: React Native" \
  -d '{"text": "My car broke down on the highway, I need a tow", "conversation_id": "test123"}'

# Process dialog with emotional content to see adapted response
curl -X POST http://localhost:8001/api/dialog \
  -H "Content-Type: application/json" \
  -H "X-Platform: React Native" \
  -d '{"text": "This is an emergency! My car broke down on the highway and I need urgent help!", "conversation_id": "test123"}'
```

## React Native Integration

### 1. Create API Service Module

Create a dedicated service module to handle API calls in your React Native project:

```javascript
// src/services/chatbotService.js
const API_BASE_URL = 'http://localhost:8001/api' // Updated to use port 8001
const API_NLU_URL = `${API_BASE_URL}/nlu`
const API_DIALOG_URL = `${API_BASE_URL}/dialog`
const API_HEALTH_URL = `${API_BASE_URL}/health`

/**
 * Process text through NLU only (no conversation state)
 * @param {string} text - User message to analyze
 * @returns {Promise<Object>} - NLU result with intent, entities, and sentiment
 */
export const processTextNLU = async (text) => {
  try {
    const response = await fetch(API_NLU_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Platform': 'React Native',
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
      sentiment: { label: 'neutral', score: 0.5 },
    }
  }
}

/**
 * Process dialog with conversation state
 * @param {string} text - User message
 * @param {string} conversationId - Optional conversation ID for maintaining context
 * @returns {Promise<Object>} - Bot response and conversation ID
 */
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
        'X-Platform': 'React Native',
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

/**
 * Check if API is available
 * @returns {Promise<boolean>} - True if API is healthy
 */
export const checkApiHealth = async () => {
  try {
    const response = await fetch(API_HEALTH_URL, {
      method: 'GET',
      headers: {
        'X-Platform': 'React Native',
      },
    })
    return response.ok
  } catch (error) {
    console.error('API health check failed:', error)
    return false
  }
}

/**
 * Initialize conversation with API
 * Performs health check and returns a new conversation ID
 * @returns {Promise<string|null>} - New conversation ID or null if API unavailable
 */
export const initializeConversation = async () => {
  try {
    // First check API health
    const isHealthy = await checkApiHealth()
    if (!isHealthy) {
      console.error('API is not healthy, cannot initialize conversation')
      return null
    }

    // Start a conversation with a greeting
    const result = await processDialog('Hello', null)
    return result.conversation_id
  } catch (error) {
    console.error('Failed to initialize conversation:', error)
    return null
  }
}
```

### 2. Chat Component Implementation

Here's a complete Chat component that integrates with the API service:

```javascript
// src/components/Chat.js
import React, { useState, useEffect, useRef } from 'react'
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
} from 'react-native'
import {
  processDialog,
  checkApiHealth,
  initializeConversation,
} from '../services/chatbotService'

const Chat = () => {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiAvailable, setApiAvailable] = useState(true)
  const [conversationId, setConversationId] = useState(null)
  const [isInitializing, setIsInitializing] = useState(true)

  const flatListRef = useRef(null)

  // Initialize conversation and check API health
  useEffect(() => {
    const initialize = async () => {
      setIsInitializing(true)

      // Check API health
      const isHealthy = await checkApiHealth()
      setApiAvailable(isHealthy)

      if (isHealthy) {
        // Initialize a conversation
        const newConversationId = await initializeConversation()
        if (newConversationId) {
          setConversationId(newConversationId)

          // Add welcome message
          setMessages([
            {
              id: Date.now().toString(),
              text: 'Welcome! How can I assist you with your vehicle today?',
              sender: 'bot',
            },
          ])
        }
      }

      setIsInitializing(false)
    }

    initialize()
  }, [])

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (messages.length > 0 && flatListRef.current) {
      setTimeout(() => {
        flatListRef.current.scrollToEnd({ animated: true })
      }, 100)
    }
  }, [messages])

  const handleSend = async () => {
    if (!inputText.trim() || !apiAvailable || isLoading) return

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
      if (
        result.conversation_id &&
        (!conversationId || conversationId === 'error')
      ) {
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

      // Check if API is still available
      const isHealthy = await checkApiHealth()
      setApiAvailable(isHealthy)
    } finally {
      setIsLoading(false)
    }
  }

  // Retry connecting to API
  const handleRetryConnection = async () => {
    const isHealthy = await checkApiHealth()
    setApiAvailable(isHealthy)

    if (isHealthy && !conversationId) {
      const newConversationId = await initializeConversation()
      if (newConversationId) {
        setConversationId(newConversationId)
      }
    }
  }

  if (isInitializing) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0084ff" />
        <Text style={styles.loadingText}>Connecting to chatbot...</Text>
      </View>
    )
  }

  return (
    <SafeAreaView style={styles.container}>
      {!apiAvailable && (
        <View style={styles.apiWarning}>
          <Text style={styles.apiWarningText}>
            Cannot connect to the chatbot API. Please check your connection.
          </Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={handleRetryConnection}
          >
            <Text style={styles.retryButtonText}>Retry Connection</Text>
          </TouchableOpacity>
        </View>
      )}

      <FlatList
        ref={flatListRef}
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
            <Text
              style={[
                styles.messageText,
                item.sender === 'user'
                  ? styles.userMessageText
                  : styles.botMessageText,
              ]}
            >
              {item.text}
            </Text>
          </View>
        )}
        style={styles.messageList}
        contentContainerStyle={styles.messageListContent}
      />

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 0}
      >
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type a message..."
            placeholderTextColor="#999"
            onSubmitEditing={handleSend}
            editable={!isLoading && apiAvailable}
            multiline
          />
          {isLoading ? (
            <ActivityIndicator
              size="small"
              color="#0084ff"
              style={styles.sendButton}
            />
          ) : (
            <TouchableOpacity
              style={[
                styles.sendButton,
                !apiAvailable && styles.disabledButton,
              ]}
              onPress={handleSend}
              disabled={!apiAvailable || !inputText.trim()}
            >
              <Text style={styles.sendButtonText}>Send</Text>
            </TouchableOpacity>
          )}
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  apiWarning: {
    backgroundColor: '#ffcccc',
    padding: 15,
    alignItems: 'center',
  },
  apiWarningText: {
    color: '#cc0000',
    fontSize: 14,
    marginBottom: 8,
  },
  retryButton: {
    backgroundColor: '#cc0000',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  retryButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  messageList: {
    flex: 1,
  },
  messageListContent: {
    padding: 10,
    paddingBottom: 15,
  },
  messageBubble: {
    borderRadius: 20,
    padding: 15,
    marginVertical: 5,
    maxWidth: '80%',
    minWidth: 100,
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
  },
  userMessageText: {
    color: 'white',
  },
  botMessageText: {
    color: '#333',
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
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#0084ff',
    borderRadius: 20,
    width: 60,
    height: 45,
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

### 3. Adding the Chat Component to Your App

To add the Chat component to your app, simply import it and include it in your navigation or screen:

```javascript
// App.js or your navigation component
import React from 'react'
import { StyleSheet, View, Text } from 'react-native'
import { NavigationContainer } from '@react-navigation/native'
import { createStackNavigator } from '@react-navigation/stack'
import Chat from './src/components/Chat'

const Stack = createStackNavigator()

const HomeScreen = ({ navigation }) => (
  <View style={styles.homeContainer}>
    <Text style={styles.title}>Roadside Assistance Chatbot</Text>
    <TouchableOpacity
      style={styles.chatButton}
      onPress={() => navigation.navigate('Chat')}
    >
      <Text style={styles.chatButtonText}>Start Chat</Text>
    </TouchableOpacity>
  </View>
)

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: 'Auto Assistance' }}
        />
        <Stack.Screen
          name="Chat"
          component={Chat}
          options={{ title: 'Chat with Assistant' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  )
}

const styles = StyleSheet.create({
  homeContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  chatButton: {
    backgroundColor: '#0084ff',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  chatButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
})
```

### 4. Environment Configuration

Create an environment configuration file to manage API URLs for different environments:

```javascript
// src/config/env.js
const ENV = {
  dev: {
    apiUrl: 'http://localhost:8001/api',
  },
  staging: {
    apiUrl: 'http://staging-server:8001/api',
  },
  prod: {
    apiUrl: 'https://production-server/api',
  },
}

// For React Native, determine which platform to use
const getEnvVars = (env = process.env.NODE_ENV || 'development') => {
  if (env === 'development') {
    return ENV.dev
  } else if (env === 'staging') {
    return ENV.staging
  } else if (env === 'production') {
    return ENV.prod
  }
}

export default getEnvVars
```

Then update your chatbotService.js to use this configuration:

```javascript
// src/services/chatbotService.js
import getEnvVars from '../config/env'

const { apiUrl } = getEnvVars()
const API_BASE_URL = apiUrl
const API_NLU_URL = `${API_BASE_URL}/nlu`
const API_DIALOG_URL = `${API_BASE_URL}/dialog`
const API_HEALTH_URL = `${API_BASE_URL}/health`

// ... rest of the service code
```

## Testing in Development

When testing locally:

1. **Start the local API server**:

   ```bash
   # RECOMMENDED: Use the start_api.sh script
   ./scripts/start_api.sh
   ```

2. **For iOS simulator**: Use 'http://localhost:8001/api' as the base URL
3. **For Android emulator**: Use 'http://10.0.2.2:8001/api' as the base URL
4. **For physical devices on the same WiFi**: Use your computer's local IP address: 'http://192.168.x.x:8001/api'

## Important Notes

1. **Use the scripts/start_api.sh script** - Always use the provided script to start the server for proper logging and setup
2. **Set X-Platform header** - Always include 'X-Platform: React Native' header in all API requests for proper logging
3. **Conversation State** - The dialog endpoint maintains state between turns when you provide the same conversation_id
4. **Dialog Flow** - The system will guide users through required information for different flows (towing, roadside assistance, etc.)
5. **Error Handling** - Always implement proper error handling as shown in the example
6. **Connectivity Issues** - Test your app's behavior when the API is unreachable
7. **Testing with NLU-only** - For specialized use cases, you can use the `/api/nlu` endpoint to get just the NLU results without dialog management

## Troubleshooting

### Common Issues

1. **Chat Messages Not Visible in Terminal**:

   - Ensure you started the server using `./scripts/start_api.sh` (not directly with Python)
   - Verify your app is sending the 'X-Platform: React Native' header with every request
   - Make sure you're looking at the right terminal window where the server is running

2. **Connection Refused**: Make sure the API server is running on port 8001 before attempting to connect
3. **Android Emulator Connection**: Remember that Android emulators can't access `localhost` directly - use `10.0.2.2` instead
4. **Slow Responses**: The NLU model processing might take a moment, especially on the first request - implement proper loading states
5. **Missing Conversation ID**: Always store and reuse the conversation_id returned from the API to maintain context
6. **Conversation Timeout**: If your system doesn't receive messages for a long time, the conversation context might be lost
