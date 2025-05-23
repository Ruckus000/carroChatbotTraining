# ngrok Setup Guide for Carro Chatbot

This guide explains how to set up ngrok to create a public tunnel to your chatbot API server, allowing your React Native app to connect from any network without changing IP addresses.

## Quick Start (Recommended)

### Option 1: Simple Network Discovery Script

For the fastest setup without external dependencies, use our enhanced startup script that displays all connection options:

```bash
./scripts/start_api.sh
```

This will show you all available IP addresses to connect from different devices.

### Option 2: ngrok Public Tunnel (Best for Development)

For a permanent URL that works from any network:

## 🚀 ngrok Setup Instructions

### Step 1: Create ngrok Account (Free)

1. Go to [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)
2. Sign up for a free account
3. Verify your email address

### Step 2: Get Your Auth Token

1. After signing in, go to [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
2. Copy your auth token (looks like: `2abcd...xyz123`)

### Step 3: Install Auth Token

```bash
# Install your auth token (replace YOUR_TOKEN with your actual token)
ngrok config add-authtoken YOUR_TOKEN
```

### Step 4: Start Server with ngrok

```bash
# Use our enhanced startup script
./scripts/start_api_with_ngrok.sh
```

This script will:

- ✅ Start your API server on localhost:8001
- ✅ Create an ngrok tunnel with a public URL
- ✅ Display the public URL for your React Native app
- ✅ Show monitoring links

## 📱 React Native Integration

### Method 1: Using ngrok URL (Recommended)

Once you have ngrok running, update your React Native app configuration:

#### 1. Create Environment Configuration

Create or update `src/config/env.js`:

```javascript
// src/config/env.js
const ENV = {
  dev: {
    // Replace with your actual ngrok URL from the startup script
    apiUrl: 'https://abcd1234.ngrok.io/api',
  },
  staging: {
    apiUrl: 'https://staging-server.com/api',
  },
  prod: {
    apiUrl: 'https://production-server.com/api',
  },
}

const getEnvVars = (env = __DEV__ ? 'dev' : 'prod') => {
  return ENV[env] || ENV.dev
}

export default getEnvVars
```

#### 2. Update API Service

Modify your `src/services/chatbotService.js`:

```javascript
// src/services/chatbotService.js
import getEnvVars from '../config/env'

// Get the API base URL from environment config
const { apiUrl } = getEnvVars()
const API_BASE_URL = apiUrl
const API_NLU_URL = `${API_BASE_URL}/nlu`
const API_DIALOG_URL = `${API_BASE_URL}/dialog`
const API_HEALTH_URL = `${API_BASE_URL}/health`

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

// ... rest of your API functions
```

### Method 2: Dynamic IP Configuration

For more flexibility, create a settings screen in your app:

#### 1. Create Settings Screen

```javascript
// src/screens/SettingsScreen.js
import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native'
import AsyncStorage from '@react-native-async-storage/async-storage'

const SettingsScreen = () => {
  const [serverURL, setServerURL] = useState('')
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    loadSavedURL()
  }, [])

  const loadSavedURL = async () => {
    try {
      const savedURL = await AsyncStorage.getItem('apiServerURL')
      if (savedURL) {
        setServerURL(savedURL)
        testConnection(savedURL)
      }
    } catch (error) {
      console.error('Error loading saved URL:', error)
    }
  }

  const testConnection = async (url) => {
    try {
      const response = await fetch(`${url}/health`, {
        method: 'GET',
        headers: { 'X-Platform': 'React Native' },
      })
      setIsConnected(response.ok)
    } catch (error) {
      setIsConnected(false)
    }
  }

  const saveURL = async () => {
    try {
      let cleanURL = serverURL.trim()
      if (!cleanURL.startsWith('http')) {
        cleanURL = `http://${cleanURL}`
      }
      if (!cleanURL.endsWith('/api')) {
        cleanURL = `${cleanURL}/api`
      }

      await AsyncStorage.setItem('apiServerURL', cleanURL)
      await testConnection(cleanURL)
      Alert.alert(
        'Success',
        `Server URL saved and ${
          isConnected ? 'connected' : 'connection failed'
        }`
      )
    } catch (error) {
      Alert.alert('Error', 'Failed to save server URL')
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>API Server Configuration</Text>

      <Text style={styles.label}>Server URL:</Text>
      <TextInput
        style={styles.input}
        value={serverURL}
        onChangeText={setServerURL}
        placeholder="localhost:8001 or https://abc123.ngrok.io"
        autoCapitalize="none"
        autoCorrect={false}
      />

      <Text style={styles.helper}>
        Examples:{'\n'}• localhost:8001 (local development){'\n'}•
        192.168.1.100:8001 (local WiFi){'\n'}• https://abc123.ngrok.io (ngrok tunnel)
      </Text>

      <TouchableOpacity style={styles.button} onPress={saveURL}>
        <Text style={styles.buttonText}>Save & Test Connection</Text>
      </TouchableOpacity>

      <View
        style={[
          styles.status,
          isConnected ? styles.connected : styles.disconnected,
        ]}
      >
        <Text style={styles.statusText}>
          {isConnected ? '✅ Connected' : '❌ Not Connected'}
        </Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
    textAlign: 'center',
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: 'white',
    marginBottom: 10,
  },
  helper: {
    fontSize: 14,
    color: '#666',
    marginBottom: 20,
    fontStyle: 'italic',
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  status: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  connected: {
    backgroundColor: '#d4edda',
  },
  disconnected: {
    backgroundColor: '#f8d7da',
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
  },
})

export default SettingsScreen
```

#### 2. Update API Service for Dynamic URLs

```javascript
// src/services/chatbotService.js
import AsyncStorage from '@react-native-async-storage/async-storage'

const getApiBaseUrl = async () => {
  try {
    const savedURL = await AsyncStorage.getItem('apiServerURL')
    return savedURL || 'http://localhost:8001/api'
  } catch (error) {
    console.error('Error getting API URL:', error)
    return 'http://localhost:8001/api'
  }
}

export const processDialog = async (text, conversationId = null) => {
  try {
    const API_BASE_URL = await getApiBaseUrl()
    const API_DIALOG_URL = `${API_BASE_URL}/dialog`

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

// Similar updates for other API functions...
```

## 🔧 Available Scripts

### Start API Only

```bash
./scripts/start_api.sh
```

### Start API with ngrok

```bash
./scripts/start_api_with_ngrok.sh
```

## 📋 Connection Options Summary

| Method           | Use Case                    | Pros                                                 | Cons                                     |
| ---------------- | --------------------------- | ---------------------------------------------------- | ---------------------------------------- |
| **ngrok tunnel** | Development across networks | Works anywhere, HTTPS, persistent URL during session | Requires account, URL changes on restart |
| **Local IP**     | Same WiFi network           | Fast, direct connection                              | IP changes with networks                 |
| **localhost**    | Simulator only              | Simplest setup                                       | Only works with simulators               |

## 🔍 Troubleshooting

### ngrok Issues

- **Authentication Error**: Make sure you've added your auth token with `ngrok config add-authtoken YOUR_TOKEN`
- **Port Already in Use**: Stop existing servers with `pkill -f "python.*api.py" && pkill ngrok`
- **Can't Access Web Interface**: Go to [http://localhost:4040](http://localhost:4040) to monitor ngrok

### API Connection Issues

- **Connection Refused**: Make sure API server is running on port 8001
- **404 Errors**: Ensure you're using `/api` in your URLs
- **CORS Issues**: The API is configured to allow all origins

### React Native Issues

- **Metro Connection**: For physical devices, use your computer's IP address
- **Android Emulator**: Use `10.0.2.2:8001` instead of `localhost:8001`
- **iOS Simulator**: Use `localhost:8001` or `127.0.0.1:8001`

## 🎯 Recommended Workflow

1. **Development**: Use ngrok for consistent URLs across all devices
2. **Testing**: Use local IP addresses for faster connections
3. **Production**: Deploy to a proper hosting service

## 📞 Support

If you encounter issues:

1. Check the API server logs: `tail -f api_server.log`
2. Check ngrok logs: `tail -f ngrok.log`
3. Test API directly: `curl http://localhost:8001/api/health`
4. Verify ngrok status: Visit [http://localhost:4040](http://localhost:4040)
