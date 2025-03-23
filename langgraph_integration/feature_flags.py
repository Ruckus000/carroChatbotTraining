import os
import json
from typing import Dict, Any, Optional

class FeatureFlags:
    """
    Feature flag management for controlling LangGraph integration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "FEATURE_FLAG_CONFIG", 
            "./langgraph_integration/config/feature_flags.json"
        )
        self.flags = self._load_flags()
        
    def _load_flags(self) -> Dict[str, bool]:
        """Load feature flags from file or use defaults"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading feature flags: {e}")
        
        # Default flags - all disabled initially
        return {
            "use_langgraph": False,        # Use LangGraph for orchestration
            "use_mistral": False,          # Use Mistral for enhanced NLU
            "hybrid_detection": False,     # Use hybrid rule/ML detection
            "enhanced_logging": True,      # Use enhanced logging and metrics
            "retain_entities": True,       # Keep entity values across context switches
            "async_processing": False      # Use async processing (CPU optimization)
        }
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled"""
        return self.flags.get(flag_name, False)
    
    def enable(self, flag_name: str) -> None:
        """Enable a feature flag"""
        self.flags[flag_name] = True
        self._save_flags()
    
    def disable(self, flag_name: str) -> None:
        """Disable a feature flag"""
        self.flags[flag_name] = False
        self._save_flags()
    
    def _save_flags(self) -> None:
        """Save current flags to config file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.flags, f, indent=2)
        except Exception as e:
            print(f"Error saving feature flags: {e}") 