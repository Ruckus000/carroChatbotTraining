// ============================================================================
// REACT NATIVE INTEGRATION EXAMPLES FOR CARRO CHATBOT API
// ============================================================================

// -----------------------------------------------------------------------------
// Option 1: Environment-based Configuration (Recommended)
// -----------------------------------------------------------------------------

// File: src/config/env.js
const ENV = {
  dev: {
    // Choose one of these based on your setup:
    apiUrl: 'http://localhost:8001/api',           // iOS Simulator
    // apiUrl: 'http://10.0.2.2:8001/api',        // Android Emulator  
    // apiUrl: 'http://192.168.1.100:8001/api',   // Physical device (replace with your IP)
    // apiUrl: 'https://abc123.ngrok.io/api',     // ngrok tunnel (any network)
  },
  staging: {
    apiUrl: 'https://staging-server.com/api',
  },
  prod: {
    apiUrl: 'https://production-server.com/api',
  },
};

const getEnvVars = (env = __DEV__ ? 'dev' : 'prod') => {
  return ENV[env] || ENV.dev;
};

export default getEnvVars;

// -----------------------------------------------------------------------------
// Option 2: Chat Service with Environment Configuration
// -----------------------------------------------------------------------------

// File: src/services/chatbotService.js
import getEnvVars from '../config/env';

const { apiUrl } = getEnvVars();
const API_BASE_URL = apiUrl;

export const processDialog = async (text, conversationId = null) => {
  try {
    const body = { text };
    if (conversationId) {
      body.conversation_id = conversationId;
    }

    const response = await fetch(`${API_BASE_URL}/dialog`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Platform': 'React Native',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing dialog:', error);
    return {
      text: 'Sorry, I encountered an error processing your request.',
      conversation_id: conversationId || 'error',
    };
  }
};

export const processNLU = async (text) => {
  try {
    const response = await fetch(`${API_BASE_URL}/nlu`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Platform': 'React Native',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing NLU:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'X-Platform': 'React Native',
      },
    });

    return response.ok;
  } catch (error) {
    console.error('Error checking health:', error);
    return false;
  }
};

// -----------------------------------------------------------------------------
// Option 3: Dynamic Configuration with Settings Screen
// -----------------------------------------------------------------------------

// File: src/services/dynamicChatbotService.js
import AsyncStorage from '@react-native-async-storage/async-storage';

const getApiBaseUrl = async () => {
  try {
    const savedURL = await AsyncStorage.getItem('apiServerURL');
    return savedURL || 'http://localhost:8001/api';
  } catch (error) {
    console.error('Error getting API URL:', error);
    return 'http://localhost:8001/api';
  }
};

export const processDialog = async (text, conversationId = null) => {
  try {
    const API_BASE_URL = await getApiBaseUrl();
    const body = { text };
    if (conversationId) {
      body.conversation_id = conversationId;
    }

    const response = await fetch(`${API_BASE_URL}/dialog`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Platform': 'React Native',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing dialog:', error);
    return {
      text: 'Sorry, I encountered an error processing your request.',
      conversation_id: conversationId || 'error',
    };
  }
};

// -----------------------------------------------------------------------------
// Simple Chat Component Example
// -----------------------------------------------------------------------------

// File: src/components/ChatScreen.js
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
} from 'react-native';
import { processDialog, checkHealth } from '../services/chatbotService';

const ChatScreen = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    checkConnection();
    // Add initial bot message
    setMessages([
      {
        id: 1,
        text: 'Hi! I\'m here to help with vehicle services like towing, roadside assistance, and service appointments. What do you need help with?',
        isBot: true,
        timestamp: new Date(),
      },
    ]);
  }, []);

  const checkConnection = async () => {
    const connected = await checkHealth();
    setIsConnected(connected);
    if (!connected) {
      Alert.alert(
        'Connection Error',
        'Unable to connect to the API server. Please check if the server is running.'
      );
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      text: inputText,
      isBot: false,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');

    try {
      const response = await processDialog(inputText, conversationId);
      
      if (!conversationId && response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      const botMessage = {
        id: messages.length + 2,
        text: response.text,
        isBot: true,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: messages.length + 2,
        text: 'Sorry, I encountered an error. Please try again.',
        isBot: true,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Carro Assistant</Text>
        <View style={[styles.status, isConnected ? styles.connected : styles.disconnected]}>
          <Text style={styles.statusText}>
            {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </Text>
        </View>
      </View>

      <ScrollView style={styles.messagesContainer}>
        {messages.map(message => (
          <View
            key={message.id}
            style={[
              styles.message,
              message.isBot ? styles.botMessage : styles.userMessage,
            ]}
          >
            <Text style={styles.messageText}>{message.text}</Text>
          </View>
        ))}
      </ScrollView>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type your message..."
          multiline
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#007AFF',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  status: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  connected: {
    backgroundColor: '#34C759',
  },
  disconnected: {
    backgroundColor: '#FF3B30',
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  messagesContainer: {
    flex: 1,
    padding: 16,
  },
  message: {
    marginBottom: 12,
    padding: 12,
    borderRadius: 12,
    maxWidth: '80%',
  },
  botMessage: {
    backgroundColor: '#E5E5EA',
    alignSelf: 'flex-start',
  },
  userMessage: {
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
  },
  messageText: {
    fontSize: 16,
    color: '#000',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    borderRadius: 20,
    paddingHorizontal: 20,
    paddingVertical: 8,
    justifyContent: 'center',
  },
  sendButtonText: {
    color: 'white',
    fontWeight: '600',
  },
});

export default ChatScreen;

// -----------------------------------------------------------------------------
// Installation Dependencies
// -----------------------------------------------------------------------------
/*
Add these to your React Native project:

npm install @react-native-async-storage/async-storage

For iOS:
cd ios && pod install
*/ 