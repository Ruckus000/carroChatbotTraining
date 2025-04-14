// N8N Context-Aware Chatbot Code Node (Positioned after AI Agent)
// This code processes both the user message and AI response to maintain conversation context,
// now extended with bookingState to manage a multi-step booking process.

async function processMessage(items) {
    const returnItems = [];
    for (const item of items) {
        try {
            const aiResponse = item.json.text || item.json.response || item.json.content || "";
            const currentMessage = item.json.input || item.json.message || item.json.content || item.json.data?.text || "";
            let context = item.json.memory?.context || initializeContext();
            const sessionId = item.json.conversationId || item.json.sessionId || context.conversationId || generateId();
            context.conversationId = sessionId;
            const result = processMessageWithContext(currentMessage, context);
            if (aiResponse && result.context.messages) {
                if (aiResponse?.trim()) {
                    result.context.messages.push({
                        role: "assistant",
                        content: aiResponse,
                        timestamp: new Date().toISOString()
                    });
                }
            }
            returnItems.push({
                json: {
                    message: currentMessage,
                    sessionId,
                    aiResponse,
                    context: result.context,
                    detectedIntent: result.intent,
                    flow: result.flow,
                    detectedEntities: result.entities,
                    detectedNegation: result.negation,
                    detectedContextSwitch: result.contextSwitch,
                    detectedContradiction: result.contradiction,
                    confidenceScores: result.confidenceScores,
                    needsClarification: result.needsClarification,
                    processingResult: result
                }
            });
        } catch (error) {
            returnItems.push({ json: { ...item.json, error: error.message, processingFailed: true } });
        }
    }
    return returnItems;
}

function initializeContext() {
    return {
        conversationId: generateId(),
        turnCount: 0,
        flow: "unknown",
        previousIntents: [],
        entities: {},
        contextSwitches: [],
        negations: [],
        contradictions: [],
        messages: [],
        // Extended booking state initialization
        bookingState: {
            currentStep: "NotStarted", // Possible steps: "NotStarted", "Gather Vehicle Information", etc.
            bookingFlow: null, // "Towing", "Roadside Assistance", or "Service Appointment"
            bookingDetails: {}, // To store details like vehicle info, location, etc.
            confirmed: false // True when booking is finalized
        }
    };
}

function generateId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

function processMessageWithContext(text, context) {
    const updatedContext = { ...context };
    updatedContext.turnCount = (updatedContext.turnCount || 0) + 1;
    if (!updatedContext.messages) updatedContext.messages = [];
    updatedContext.messages.push({ role: "user", content: text, timestamp: new Date().toISOString() });

    // Ensure bookingState exists
    if (!updatedContext.bookingState) {
        updatedContext.bookingState = {
            currentStep: "NotStarted",
            bookingFlow: null,
            bookingDetails: {},
            confirmed: false
        };
    }

    const negationResult = detectNegation(text, updatedContext);
    const contextSwitchResult = detectContextSwitch(text, updatedContext);
    const intentResult = detectIntent(text, updatedContext);

    let flow = updatedContext.flow || "unknown";
    if (contextSwitchResult.hasContextSwitch && contextSwitchResult.newContext) {
        flow = contextSwitchResult.newContext;
        updatedContext.contextSwitches.push({
            turn: updatedContext.turnCount,
            previousFlow: updatedContext.flow,
            newFlow: flow,
            confidence: contextSwitchResult.confidence,
            timestamp: new Date().toISOString()
        });
    } else if (intentResult.intent !== "unknown" && intentResult.confidence > 0.6) {
        flow = mapIntentToFlow(intentResult.intent);
    }
    updatedContext.flow = flow;

    if (!updatedContext.previousIntents) updatedContext.previousIntents = [];
    updatedContext.previousIntents.push({
        intent: intentResult.intent,
        confidence: intentResult.confidence,
        turn: updatedContext.turnCount,
        timestamp: new Date().toISOString()
    });

    const entities = extractEntities(text, updatedContext);
    const contradictionResult = detectContradictions(entities, updatedContext);
    for (const [entityType, value] of Object.entries(entities)) {
        if (!updatedContext.entities) updatedContext.entities = {};
        const previousValue = updatedContext.entities[entityType];
        updatedContext.entities[entityType] = {
            value,
            previousValue: previousValue?.value,
            turn: updatedContext.turnCount,
            timestamp: new Date().toISOString()
        };
    }

    const needsClarification = intentResult.intent === "unknown" || intentResult.confidence < 0.4 || (flow === "unknown" && updatedContext.turnCount === 1);
    if (negationResult.isNegation) {
        if (!updatedContext.negations) updatedContext.negations = [];
        updatedContext.negations.push({
            turn: updatedContext.turnCount,
            confidence: negationResult.confidence,
            timestamp: new Date().toISOString()
        });
    }
    if (contradictionResult.hasContradiction) {
        if (!updatedContext.contradictions) updatedContext.contradictions = [];
        updatedContext.contradictions.push({
            turn: updatedContext.turnCount,
            entityType: contradictionResult.entityType,
            oldValue: contradictionResult.oldValue,
            newValue: contradictionResult.newValue,
            confidence: contradictionResult.confidence,
            timestamp: new Date().toISOString()
        });
    }

    const confidenceScores = {
        intent: intentResult.confidence,
        negation: negationResult.confidence,
        contextSwitch: contextSwitchResult.confidence
    };

    // --- Booking Process Integration ---
    // Step 1: Initialize booking if user mentions "book" and no bookingFlow is set.
    if (text.toLowerCase().includes("book") && !updatedContext.bookingState.bookingFlow) {
        if (text.toLowerCase().includes("towing")) {
            updatedContext.bookingState.bookingFlow = "Towing";
            updatedContext.bookingState.currentStep = "Gather Vehicle Information";
        } else if (text.toLowerCase().includes("roadside")) {
            updatedContext.bookingState.bookingFlow = "Roadside Assistance";
            updatedContext.bookingState.currentStep = "Identify Specific Roadside Need";
        } else if (text.toLowerCase().includes("appointment")) {
            updatedContext.bookingState.bookingFlow = "Service Appointment";
            updatedContext.bookingState.currentStep = "Identify Service Type";
        }
    }

    // Towing Service Flow
    if (updatedContext.bookingState.bookingFlow === "Towing") {
        if (updatedContext.bookingState.currentStep === "Gather Vehicle Information") {
            updatedContext.bookingState.bookingDetails.vehicleInfo = text;
            updatedContext.bookingState.currentStep = "Gather Location Information";
        } else if (updatedContext.bookingState.currentStep === "Gather Location Information") {
            updatedContext.bookingState.bookingDetails.locationInfo = text;
            updatedContext.bookingState.currentStep = "Gather Situation Details";
        } else if (updatedContext.bookingState.currentStep === "Gather Situation Details") {
            updatedContext.bookingState.bookingDetails.situationDetails = text;
            updatedContext.bookingState.currentStep = "Confirm Booking Request";
        } else if (updatedContext.bookingState.currentStep === "Confirm Booking Request") {
            if (text.toLowerCase().includes("confirm")) {
                updatedContext.bookingState.confirmed = true;
            } else if (text.toLowerCase().includes("change vehicle")) {
                updatedContext.bookingState.currentStep = "Gather Vehicle Information";
            } else if (text.toLowerCase().includes("change location")) {
                updatedContext.bookingState.currentStep = "Gather Location Information";
            } else if (text.toLowerCase().includes("change situation")) {
                updatedContext.bookingState.currentStep = "Gather Situation Details";
            }
        }
    }

    // Roadside Assistance Flow
    else if (updatedContext.bookingState.bookingFlow === "Roadside Assistance") {
        if (updatedContext.bookingState.currentStep === "Identify Specific Roadside Need") {
            updatedContext.bookingState.bookingDetails.roadsideNeed = text;
            updatedContext.bookingState.currentStep = "Gather Vehicle Information";
        } else if (updatedContext.bookingState.currentStep === "Gather Vehicle Information") {
            updatedContext.bookingState.bookingDetails.vehicleInfo = text;
            updatedContext.bookingState.currentStep = "Gather Location Information";
        } else if (updatedContext.bookingState.currentStep === "Gather Location Information") {
            updatedContext.bookingState.bookingDetails.locationInfo = text;
            updatedContext.bookingState.currentStep = "Confirm Booking Request";
        } else if (updatedContext.bookingState.currentStep === "Confirm Booking Request") {
            if (text.toLowerCase().includes("confirm")) {
                updatedContext.bookingState.confirmed = true;
            } else if (text.toLowerCase().includes("change roadside")) {
                updatedContext.bookingState.currentStep = "Identify Specific Roadside Need";
            } else if (text.toLowerCase().includes("change vehicle")) {
                updatedContext.bookingState.currentStep = "Gather Vehicle Information";
            } else if (text.toLowerCase().includes("change location")) {
                updatedContext.bookingState.currentStep = "Gather Location Information";
            }
        }
    }

    // Service Appointment Flow
    else if (updatedContext.bookingState.bookingFlow === "Service Appointment") {
        if (updatedContext.bookingState.currentStep === "Identify Service Type") {
            updatedContext.bookingState.bookingDetails.serviceType = text;
            updatedContext.bookingState.currentStep = "Gather Vehicle Information";
        } else if (updatedContext.bookingState.currentStep === "Gather Vehicle Information") {
            updatedContext.bookingState.bookingDetails.vehicleInfo = text;
            updatedContext.bookingState.currentStep = "Schedule Service";
        } else if (updatedContext.bookingState.currentStep === "Schedule Service") {
            updatedContext.bookingState.bookingDetails.schedule = text;
            updatedContext.bookingState.currentStep = "Gather Additional Information";
        } else if (updatedContext.bookingState.currentStep === "Gather Additional Information") {
            updatedContext.bookingState.bookingDetails.additionalInfo = text;
            updatedContext.bookingState.currentStep = "Confirm Appointment";
        } else if (updatedContext.bookingState.currentStep === "Confirm Appointment") {
            if (text.toLowerCase().includes("confirm")) {
                updatedContext.bookingState.confirmed = true;
            } else if (text.toLowerCase().includes("change service type")) {
                updatedContext.bookingState.currentStep = "Identify Service Type";
            } else if (text.toLowerCase().includes("change vehicle")) {
                updatedContext.bookingState.currentStep = "Gather Vehicle Information";
            } else if (text.toLowerCase().includes("change schedule")) {
                updatedContext.bookingState.currentStep = "Schedule Service";
            } else if (text.toLowerCase().includes("change additional")) {
                updatedContext.bookingState.currentStep = "Gather Additional Information";
            }
        }
    }

    return {
        intent: intentResult.intent,
        flow,
        entities,
        context: updatedContext,
        negation: negationResult.isNegation,
        contextSwitch: contextSwitchResult.hasContextSwitch,
        contradiction: contradictionResult.hasContradiction,
        confidenceScores,
        needsClarification
    };
}

// Export for n8n
module.exports = {
    processMessage,
    initializeContext,
    generateId,
    processMessageWithContext
}; 