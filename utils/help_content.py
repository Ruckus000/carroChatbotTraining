"""
Help Content - Explanations and tooltips for the NLU Benchmarking Dashboard
"""

# Metric explanations
METRIC_HELP = {
    # Intent metrics
    "intent_accuracy": "Percentage of examples where the predicted intent matches the true intent.",
    "intent_precision": "Ratio of correctly predicted intents to the total predicted intents.",
    "intent_recall": "Ratio of correctly predicted intents to all actual intents.",
    "intent_f1": "Harmonic mean of precision and recall (0-1). Higher is better.",
    
    # Entity metrics
    "entity_precision": "Ratio of correctly predicted entities to total predicted entities.",
    "entity_recall": "Ratio of correctly predicted entities to all actual entities.",
    "entity_f1": "Harmonic mean of entity precision and recall.",
    
    # Error metrics
    "error_rate": "Percentage of examples with incorrect intent predictions.",
    "misclassifications": "Number of incorrect predictions out of total examples.",
    "avg_error_confidence": "Average confidence score for incorrect predictions.",
    
    # General metrics
    "support": "Number of actual occurrences of a class in the dataset.",
}

# Section help content
SECTION_HELP = {
    "intent_performance": """
    **How to interpret intent performance:**
    - Higher F1 scores (closer to 1.0) indicate better overall performance
    - Look for intents with low F1 scores for improvement
    - Precision < Recall: Model is over-predicting this intent
    - Recall < Precision: Model is missing instances of this intent
    """,
    
    "entity_performance": """
    **How to interpret entity performance:**
    - Entity extraction requires correct entity type and text span
    - "Micro avg" weights results by frequency; "Macro avg" treats all entities equally
    - Entity F1 is typically lower than intent F1 due to the task complexity
    - Low support entities may need more training examples
    """,
    
    "error_analysis": """
    **How to interpret error analysis:**
    - Error patterns show frequent misclassifications between specific intents
    - High-confidence errors indicate model confusion and should be prioritized
    - Compare confidence distributions between correct and incorrect predictions
    - Use the Error Explorer to investigate specific examples
    """,
    
    "confusion_matrix": """
    **How to interpret the confusion matrix:**
    - Diagonal cells show correct predictions (true intent = predicted intent)
    - Off-diagonal cells show misclassifications
    - Values are normalized by row (percentage of each true intent)
    - Bright off-diagonal cells highlight common confusion patterns
    """
}

def get_metric_help(metric_key):
    """Get help text for a specific metric"""
    return METRIC_HELP.get(metric_key, "No help available for this metric.")

def get_section_help(section_key):
    """Get help text for a specific dashboard section"""
    return SECTION_HELP.get(section_key, "No help available for this section.") 