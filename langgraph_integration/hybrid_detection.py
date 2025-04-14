from typing import Dict, Any, Optional
from langgraph_integration.feature_flags import FeatureFlags
from langgraph_integration.adapters import ExistingDetectionAdapter
from langgraph_integration.mistral_integration import MistralEnhancer

class HybridDetectionSystem:
    """
    Combines rule-based and ML-based detection with configurable weighting
    """
    
    def __init__(
        self,
        flags: FeatureFlags,
        existing_detector: ExistingDetectionAdapter,
        mistral_enhancer: Optional[MistralEnhancer] = None
    ):
        self.flags = flags
        self.existing_detector = existing_detector
        self.mistral_enhancer = mistral_enhancer
        self.rule_weight = 0.7  # Default to prioritizing rule-based detection
        self.ml_weight = 0.3    # Lower weight for ML initially
    
    def set_weights(self, rule_weight: float) -> None:
        """Set the weighting between rule-based and ML detection"""
        self.rule_weight = max(0.0, min(1.0, rule_weight))
        self.ml_weight = 1.0 - self.rule_weight
    
    def detect_negation(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect negation using hybrid approach
        Prioritizes rule-based methods but enhances with ML when available
        """
        # Get rule-based result
        rule_result = self.existing_detector.detect_negation(text)
        
        # If Mistral not enabled or available, just return rule-based result
        if not self.flags.is_enabled("use_mistral") or self.mistral_enhancer is None:
            return rule_result
        
        # If hybrid detection disabled, return rule-based result
        if not self.flags.is_enabled("hybrid_detection"):
            return rule_result
        
        # Get ML result
        ml_result = self.mistral_enhancer.detect_negation(text, context)
        
        # Apply weighted decision logic
        rule_confidence = rule_result.get("confidence", 0.0) * self.rule_weight
        ml_confidence = ml_result.get("confidence", 0.0) * self.ml_weight
        
        # If they agree, combine confidence
        if rule_result.get("is_negation") == ml_result.get("is_negation"):
            return {
                "is_negation": rule_result.get("is_negation"),
                "confidence": rule_confidence + ml_confidence,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": False
            }
        
        # If they disagree, go with the higher weighted confidence
        if rule_confidence >= ml_confidence:
            return {
                "is_negation": rule_result.get("is_negation"),
                "confidence": rule_confidence,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": True,
                "ml_based_decision": False
            }
        else:
            return {
                "is_negation": ml_result.get("is_negation"),
                "confidence": ml_confidence,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": True
            }
    
    def detect_context_switch(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect context switching using hybrid approach
        Implementation follows the same pattern as detect_negation
        """
        # Get rule-based result
        rule_result = self.existing_detector.detect_context_switch(text)
        
        # If Mistral not enabled or available, just return rule-based result
        if not self.flags.is_enabled("use_mistral") or self.mistral_enhancer is None:
            return rule_result
        
        # If hybrid detection disabled, return rule-based result
        if not self.flags.is_enabled("hybrid_detection"):
            return rule_result
        
        # Get ML result from Mistral
        ml_result = self.mistral_enhancer.detect_context_switch(text, context)
        
        # Apply weighted decision logic
        rule_confidence = rule_result.get("confidence", 0.0) * self.rule_weight
        ml_confidence = ml_result.get("confidence", 0.0) * self.ml_weight
        
        # If they agree, combine confidence
        if rule_result.get("has_context_switch") == ml_result.get("has_context_switch"):
            # For new_context, prefer rule-based if available, otherwise use ML
            new_context = rule_result.get("new_context") or ml_result.get("new_context")
            
            return {
                "has_context_switch": rule_result.get("has_context_switch"),
                "confidence": rule_confidence + ml_confidence,
                "new_context": new_context,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": False
            }
        
        # If they disagree, go with the higher weighted confidence
        if rule_confidence >= ml_confidence:
            return {
                "has_context_switch": rule_result.get("has_context_switch"),
                "confidence": rule_confidence,
                "new_context": rule_result.get("new_context"),
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": True,
                "ml_based_decision": False
            }
        else:
            return {
                "has_context_switch": ml_result.get("has_context_switch"),
                "confidence": ml_confidence,
                "new_context": ml_result.get("new_context"),
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": True
            }
    
    def analyze_intent(self, text: str, flow: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze intent using hybrid approach
        Implementation follows the same pattern as other hybrid methods
        """
        # Currently, we don't have a direct intent analysis in ExistingDetectionAdapter
        # So we'll process the message and extract intent
        rule_result = self.existing_detector.process_message(text, context)
        rule_intent = {
            "intent": rule_result.get("intent", "unknown"),
            "confidence": rule_result.get("confidence", 0.5) if "confidence" in rule_result else 0.5
        }
        
        # If Mistral not enabled or available, just return rule-based result
        if not self.flags.is_enabled("use_mistral") or self.mistral_enhancer is None:
            return rule_intent
        
        # If hybrid detection disabled, return rule-based result
        if not self.flags.is_enabled("hybrid_detection"):
            return rule_intent
        
        # Get ML result
        ml_intent = self.mistral_enhancer.analyze_intent(text, context)
        
        # Apply weighted decision logic
        rule_confidence = rule_intent.get("confidence", 0.0) * self.rule_weight
        ml_confidence = ml_intent.get("confidence", 0.0) * self.ml_weight
        
        # If they agree, combine confidence
        if rule_intent.get("intent") == ml_intent.get("intent"):
            return {
                "intent": rule_intent.get("intent"),
                "confidence": rule_confidence + ml_confidence,
                "rule_based": rule_intent,
                "ml_based": ml_intent,
                "rule_based_decision": False,
                "ml_based_decision": False
            }
        
        # If they disagree, go with the higher weighted confidence
        if rule_confidence >= ml_confidence:
            return {
                "intent": rule_intent.get("intent"),
                "confidence": rule_confidence,
                "rule_based": rule_intent,
                "ml_based": ml_intent,
                "rule_based_decision": True,
                "ml_based_decision": False
            }
        else:
            return {
                "intent": ml_intent.get("intent"),
                "confidence": ml_confidence,
                "rule_based": rule_intent,
                "ml_based": ml_intent,
                "rule_based_decision": False,
                "ml_based_decision": True
            } 