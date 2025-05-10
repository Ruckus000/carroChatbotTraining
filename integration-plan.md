# Updated Abbreviated Plan

Here's the ultra-minimal plan updated to account for state management, UI dependencies, and potential hidden dependencies:

**Phase 1: Simplify Configuration**

1. **Edit** Carro/src/config/env.ts to only export your NLU URL:

```typescript
// Carro/src/config/env.ts
export default {
  chatbotApiUrl: 'http://10.0.0.197:8001/api',
}
```

2. **Remove** every other setting or flag—no voice URLs, no feature flags.

✅ **Checkpoint:**
• Open env.ts and confirm it has exactly one key (chatbotApiUrl) and nothing else.
• Run console.log(env.chatbotApiUrl) in your app's entry to verify it prints only your /api URL.
• Check if any imports of env.ts are showing errors in the editor due to missing properties.

**Phase 2: Replace Your Chat Service**

1. **Overwrite** Carro/src/services/chatbotService.ts with:

```typescript
import axios from 'axios'
import env from '../config/env'

export interface DialogResponse {
  text: string
  conversation_id: string
}

export interface APIError {
  statusCode: number
  message: string
  detail?: string
}

export const sendDialogMessage = async (
  text: string,
  convId?: string
): Promise<DialogResponse> => {
  try {
    const body: any = { text }
    if (convId) body.conversation_id = convId

    console.log(`Sending to ${env.chatbotApiUrl}/dialog:`, body)

    const { data } = await axios.post<DialogResponse>(
      `${env.chatbotApiUrl}/dialog`,
      body,
      { headers: { 'Content-Type': 'application/json' } }
    )

    console.log('Response:', data)
    return data
  } catch (error) {
    console.error('Error in sendDialogMessage:', error)
    // Re-throw as a structured error object
    const apiError: APIError = {
      statusCode: error.response?.status || 500,
      message: error.message || 'Unknown error',
      detail: error.response?.data?.detail || error.toString(),
    }
    throw apiError
  }
}

export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await axios.get(`${env.chatbotApiUrl}/health`, {
      timeout: 5000,
    })
    return response.status === 200 && response.data?.status === 'healthy'
  } catch (error) {
    console.warn('API health check failed:', error)
    return false
  }
}
```

2. **Delete** any STT/TTS logic from this file.

✅ **Checkpoint:**
• From a Node REPL or Postman, send a POST to http://10.0.0.197:8001/api/dialog with { "text": "hello" } and confirm you get { text: "...", conversation_id: "..." }.
• Import sendDialogMessage in a small script and verify it returns the same.
• Verify the error handling by testing with an invalid URL.

**Phase 3: Update State Management**

1. **Examine** useResponseStore.ts and ensure it has the necessary state for text-only chat:

```typescript
// Carro/src/stores/useResponseStore.ts - Minimal Version
import { create } from 'zustand'

interface ResponseState {
  isProcessing: boolean
  error: string | null
  conversationId: string | null
  responseText: string | null

  setIsProcessing: (isProcessing: boolean) => void
  setError: (error: string | null) => void
  setConversationId: (id: string | null) => void
  setResponseText: (text: string | null) => void
  reset: () => void
}

const useResponseStore = create<ResponseState>()((set) => ({
  isProcessing: false,
  error: null,
  conversationId: null,
  responseText: null,

  setIsProcessing: (isProcessing) => set({ isProcessing }),
  setError: (error) => set({ error }),
  setConversationId: (id) => set({ conversationId: id }),
  setResponseText: (text) => set({ responseText: text }),
  reset: () =>
    set({
      isProcessing: false,
      error: null,
      conversationId: null,
      responseText: null,
    }),
}))

export default useResponseStore
```

2. **Update** useOptimizedSelectors.ts to remove voice-related selectors:

```typescript
// useOptimizedSelectors.ts - Simplified version
import { useMemo } from 'react'
import useChatStore from '../stores/useChatStore'
import useResponseStore from '../stores/useResponseStore'
import { Message } from '../types/chat'

export const useOptimizedSelectors = () => {
  // Chat store
  const messages = useChatStore((state) => state.messages)
  const addMessage = useChatStore((state) => state.addMessage)
  const updateMessageStatus = useChatStore((state) => state.updateMessageStatus)
  const addWelcomeMessage = useChatStore((state) => state.addWelcomeMessage)

  // Response store
  const {
    isProcessing,
    setIsProcessing,
    responseText,
    setResponseText,
    error,
    setError,
    conversationId,
    setConversationId,
  } = useResponseStore((state) => ({
    isProcessing: state.isProcessing,
    setIsProcessing: state.setIsProcessing,
    responseText: state.responseText,
    setResponseText: state.setResponseText,
    error: state.error,
    setError: state.setError,
    conversationId: state.conversationId,
    setConversationId: state.setConversationId,
  }))

  // Computed selectors
  const lastUserMessage = useMemo(() => {
    const userMessages = messages.filter((msg) => msg.sender === 'user')
    return userMessages.length > 0
      ? userMessages[userMessages.length - 1]
      : null
  }, [messages])

  const lastAssistantMessage = useMemo(() => {
    const assistantMessages = messages.filter(
      (msg) => msg.sender === 'assistant'
    )
    return assistantMessages.length > 0
      ? assistantMessages[assistantMessages.length - 1]
      : null
  }, [messages])

  return {
    // Chat store
    messages,
    addMessage,
    updateMessageStatus,
    addWelcomeMessage,

    // Response store
    isProcessing,
    setIsProcessing,
    responseText,
    setResponseText,
    error,
    setError,
    conversationId,
    setConversationId,

    // Computed
    lastUserMessage,
    lastAssistantMessage,
  }
}
```

✅ **Checkpoint:**
• Import useResponseStore and useOptimizedSelectors in a test file to verify they compile without errors.
• Check console for any errors regarding missing state or actions.
• Make sure no voice-related imports remain.

**Phase 4: Wire Only the Text Path in the Container**

1. **Open** CarroMergedInterfaceContainer.tsx and **update** to:

```typescript
import React, { useState, useEffect, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import {
  sendDialogMessage,
  checkApiHealth,
} from '../../services/chatbotService'
import { CarroMergedInterfaceUI } from './CarroMergedInterfaceUI'
import { useOptimizedSelectors } from '../../hooks/useOptimizedSelectors'

const isTestEnvironment = () => process.env.NODE_ENV === 'test'

export const CarroMergedInterfaceContainer: React.FC = () => {
  // Local state
  const [inputText, setInputText] = useState('')
  const [quickOptionsVisible, setQuickOptionsVisible] = useState(true)

  // Use optimized selectors - Get only what we need for text chat
  const {
    messages,
    addMessage,
    updateMessageStatus,
    addWelcomeMessage,
    isProcessing,
    setIsProcessing,
    error,
    setError,
    conversationId,
    setConversationId,
    setResponseText,
  } = useOptimizedSelectors()

  // Initialize chat and check API health
  useEffect(() => {
    const initialize = async () => {
      if (messages.length === 0 && !isTestEnvironment()) {
        const isHealthy = await checkApiHealth()
        if (!isHealthy) {
          setError(
            'Chat service is currently unavailable. Please try again later.'
          )
        }
        addWelcomeMessage()
        setConversationId(null)
        setQuickOptionsVisible(true)
      } else if (messages.length > 0) {
        setQuickOptionsVisible(false)
      }
    }
    initialize()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Process and relay a text message to the API
  const processAndRelayMessage = useCallback(
    async (textToSend: string) => {
      if (!textToSend.trim() || isProcessing) return

      // Generate a unique ID for this message
      const userMessageId = uuidv4()

      // Add the user message to the chat with 'sending' status
      addMessage({
        id: userMessageId,
        text: textToSend,
        sender: 'user',
        status: 'sending',
        type: 'text',
      })

      // Clear input and hide quick options
      setInputText('')
      setQuickOptionsVisible(false)
      setIsProcessing(true)
      setError(null)

      try {
        // Send the message to the API
        const response = await sendDialogMessage(textToSend, conversationId)

        // Update the user message status to 'sent'
        updateMessageStatus(userMessageId, 'sent')

        // Add the bot's response
        addMessage({
          text: response.text,
          sender: 'assistant',
          status: 'sent',
          type: 'text',
        })

        // Store the conversation ID for continuity
        setConversationId(response.conversation_id)
        setResponseText(response.text)
      } catch (apiError: any) {
        console.error('Error in processAndRelayMessage:', apiError)

        // Update user message status to show the error
        updateMessageStatus(userMessageId, 'error')

        // Set the error message and show it in the chat
        const errorMessage =
          apiError.detail || apiError.message || 'Failed to get response'
        setError(errorMessage)

        // Add an error message to the chat
        addMessage({
          text: `Error: ${errorMessage}`,
          sender: 'assistant',
          status: 'error',
          type: 'text',
        })
      } finally {
        setIsProcessing(false)
      }
    },
    [
      isProcessing,
      conversationId,
      addMessage,
      updateMessageStatus,
      setIsProcessing,
      setError,
      setConversationId,
      setResponseText,
    ]
  )

  // Handler for text input send button
  const handleSendMessage = useCallback(() => {
    processAndRelayMessage(inputText)
  }, [inputText, processAndRelayMessage])

  // Handler for quick option selection
  const handleQuickOption = useCallback(
    (message: string) => {
      processAndRelayMessage(message)
    },
    [processAndRelayMessage]
  )

  return (
    <CarroMergedInterfaceUI
      inputText={inputText}
      setInputText={setInputText}
      messages={messages}
      isProcessing={isProcessing}
      handleSendTextMessage={handleSendMessage}
      handleQuickOption={handleQuickOption}
      quickOptionsVisible={quickOptionsVisible}
      error={error}
    />
  )
}
```

2. **Remove** all voice-related imports, state, and functions.

✅ **Checkpoint:**
• Build the app (npm run ios or npm run android)—it should compile with zero missing imports.
• Check console for any warnings about missing props or dependencies.
• Observe the container's behavior to ensure state updates correctly (message sent → status changes → response appears).

**Phase 5: Simplify the UI**

1. **Open** CarroMergedInterfaceUI.tsx and **update** the props interface:

```typescript
interface CarroMergedInterfaceUIProps {
  inputText: string
  setInputText: (text: string) => void
  messages: Message[]
  isProcessing: boolean
  handleSendTextMessage: () => void
  handleQuickOption: (message: string) => void
  quickOptionsVisible?: boolean
  error?: string | null
}
```

2. **Simplify** the JSX to remove all voice UI components:

```tsx
<Animated.View style={styles.container}>
  {/* Header */}
  <Animated.View style={styles.header} testID="chat-header">
    <Text style={styles.headerTitle}>Carro Assistant</Text>
    <View style={styles.userIconContainer}>
      <User size={18} color={theme.colors.text.secondary} />
    </View>
  </Animated.View>

  {/* Quick Options */}
  {quickOptionsVisible && (
    <Animated.View
      style={styles.quickOptionsContainer}
      testID="quick-options-section"
    >
      <QuickOptionsBar
        options={quickOptionsData}
        onSelect={handleQuickOption}
        isVisible={quickOptionsVisible}
        testID="quick-options-bar"
      />
    </Animated.View>
  )}

  {/* Messages */}
  <View style={styles.messagesContainer} testID="messages-container">
    <ChatMessageList messages={messages} />
    {isProcessing && !error && (
      <Animated.View
        style={styles.typingIndicator}
        testID="typing-indicator-container"
      >
        {/* Typing indicator dots */}
      </Animated.View>
    )}
  </View>

  {/* Input Area */}
  <Animated.View style={styles.controlsContainer} testID="input-controls">
    {/* Error Display */}
    {error && (
      <View style={styles.errorDisplayContainer}>
        <Text style={styles.errorDisplayText}>{error}</Text>
      </View>
    )}

    {/* Text Input */}
    <View style={styles.inputArea}>
      <View style={styles.textInputFullWidth}>
        <ChatInputNew
          inputText={inputText}
          setInputText={setInputText}
          isProcessing={isProcessing}
          onSendMessage={handleSendTextMessage}
        />
      </View>
    </View>
  </Animated.View>
</Animated.View>
```

3. **Update** the styles to ensure proper layout:

```typescript
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background },
  header: {
    height: 60,
    backgroundColor: theme.colors.surface,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: theme.spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border,
    // shadows...
  },
  headerTitle: {
    fontSize: theme.typography.fontSize.lg,
    fontWeight: 'bold',
    color: theme.colors.text.primary,
  },
  userIconContainer: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.subtle.primary,
  },
  quickOptionsContainer: {
    paddingVertical: theme.spacing.md,
  },
  messagesContainer: {
    flex: 1,
    paddingHorizontal: theme.spacing.md,
  },
  typingIndicator: {
    flexDirection: 'row',
    padding: theme.spacing.sm,
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.subtle.secondary,
    marginRight: 4,
  },
  controlsContainer: {
    borderTopWidth: 1,
    borderTopColor: theme.colors.border,
    backgroundColor: theme.colors.background,
    paddingBottom: Platform.OS === 'ios' ? theme.spacing.xl : theme.spacing.md, // Extra bottom padding for iOS
  },
  errorDisplayContainer: {
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.lg,
    backgroundColor: theme.colors.subtle.error,
    alignItems: 'center',
  },
  errorDisplayText: {
    color: theme.colors.status.error,
    fontSize: theme.typography.fontSize.sm,
    textAlign: 'center',
  },
  inputArea: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
  },
  textInputFullWidth: {
    flex: 1,
  },
  // Remove all voice-specific styles
})
```

✅ **Checkpoint:**
• Launch the UI in your simulator/device: verify **no** mic icon or toggle appears anywhere.
• Inspect the rendered tree (React DevTools): confirm the only input is ChatInputNew.
• Test the UI layout on different screen sizes to ensure no unexpected gaps.
• Verify that animations and transitions still work properly.

**Phase 6: Clean Up Dependencies and Test Full Flow**

1. **Check and clean up imports** across all modified files, removing any unused imports.

2. **Search** for any references to voice, audio, or recording in your codebase to identify other affected components:

```
grep -r "voice\|audio\|recording\|speech\|transcrib" ./Carro/src
```

3. **Remove** any dependencies on voice-related functionality in other components.

4. **Update package.json** to remove unused voice-related dependencies, if they won't be used anywhere else:

```bash
npm uninstall expo-av @react-native-voice/voice # If no other component uses these
```

✅ **Checkpoint:**
• Run linter to check for unused imports or variables: `npx eslint ./Carro/src/**/*.{ts,tsx}`
• Build the app and check for any warnings about missing dependencies.
• Test the complete user flow from start to finish:

- App loads and displays welcome message
- User types a message and sends it
- User message appears in chat
- Bot responds correctly
- Conversation continues with proper context
- Error states are handled gracefully

**Phase 7: Final Smoke Test**

1. **Run** your external API: `python api.py`.
2. **Start** the RN app: `npm start`.
3. **In the simulator/device**:
   • Type "ping" → tap Send
   • Verify Bot replies with text from your NLU.
   • Test a multi-turn conversation to verify the conversation_id is working.
   • Force an error (e.g., stop the API while sending) to test error handling.

✅ **Final Checkpoint:**
• Document the actual request/response in a scratch file:

```
User: ping
Bot: pong (or whatever your API returns)
```

• Confirm all UI elements are properly positioned and styled.
• Verify state management works correctly for all interactions.
• Confirm no voice-related code is being executed.

Once all checkpoints pass, you have a solid text-only integration with your external NLU API. The app will be simpler, more focused, and ready for future enhancements.
