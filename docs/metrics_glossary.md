# NLU Metrics Glossary

This glossary provides detailed explanations of all metrics used in the NLU Benchmarking Dashboard, including formulas, interpretation guidelines, and appropriate use cases.

## Table of Contents

- [Intent Classification Metrics](#intent-classification-metrics)
- [Entity Recognition Metrics](#entity-recognition-metrics)
- [Confidence Metrics](#confidence-metrics)
- [Aggregate Metrics](#aggregate-metrics)
- [Custom and Advanced Metrics](#custom-and-advanced-metrics)

## Intent Classification Metrics

### Accuracy

**Definition**: The proportion of all predictions that were correct.

**Formula**: $\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$

**Interpretation**:

- Range: 0.0 to 1.0 (or 0% to 100%)
- Higher is better
- A score of 0.85 means 85% of all intent predictions were correct

**When to Use**:

- Good for balanced datasets where all intents have similar numbers of examples
- Best as a quick, overall assessment

**Limitations**:

- Can be misleading for imbalanced datasets
- Doesn't distinguish between different types of errors

### Precision

**Definition**: The proportion of positive predictions that were correct.

**Formula**: $\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Higher values indicate fewer false positives
- A precision of 0.9 means that 90% of examples predicted as a specific intent actually belonged to that intent

**When to Use**:

- When false positives are costly
- When you want to ensure high confidence in positive predictions

**Limitations**:

- Can be artificially high if the model rarely predicts the positive class

### Recall

**Definition**: The proportion of actual positives that were correctly identified.

**Formula**: $\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Higher values indicate fewer false negatives
- A recall of 0.85 means the model correctly identified 85% of all examples of a specific intent

**When to Use**:

- When false negatives are costly
- When you need to capture as many positive instances as possible

**Limitations**:

- Can be artificially high if the model frequently predicts the positive class

### F1 Score

**Definition**: The harmonic mean of precision and recall.

**Formula**: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Balances precision and recall
- A high F1 score indicates a good balance between precision and recall

**When to Use**:

- When you need a balance between precision and recall
- When working with imbalanced datasets
- When both false positives and false negatives are important

**Limitations**:

- Doesn't allow for weighing precision and recall differently

### Support

**Definition**: The number of actual occurrences of a specific intent in the dataset.

**Formula**: Count of examples for each intent

**Interpretation**:

- Raw count (not a ratio)
- Higher numbers indicate more examples
- Important for understanding dataset balance

**When to Use**:

- To assess whether performance metrics are reliable (metrics based on few examples may be unreliable)
- To identify potential data imbalance issues

## Entity Recognition Metrics

### Entity Precision

**Definition**: The proportion of predicted entities that match the true entities exactly.

**Formula**: $\text{Entity Precision} = \frac{\text{Correctly Predicted Entities}}{\text{Total Predicted Entities}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Higher values indicate fewer spurious entities
- Entity precision considers both the entity type and the exact text span

**When to Use**:

- When you want to minimize spurious entity detections
- When entity prediction accuracy is critical

### Entity Recall

**Definition**: The proportion of actual entities that were correctly predicted.

**Formula**: $\text{Entity Recall} = \frac{\text{Correctly Predicted Entities}}{\text{Total Actual Entities}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Higher values indicate fewer missed entities
- A recall of 0.75 means the model found 75% of all entities in the text

**When to Use**:

- When you need to capture as many entities as possible
- When missing entities is more problematic than spurious detections

### Entity F1 Score

**Definition**: The harmonic mean of entity precision and recall.

**Formula**: $\text{Entity F1} = 2 \cdot \frac{\text{Entity Precision} \cdot \text{Entity Recall}}{\text{Entity Precision} + \text{Entity Recall}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Balances entity precision and recall
- The best single metric for entity recognition performance

**When to Use**:

- As the primary metric for entity recognition evaluation
- When both missing entities and spurious detections are important

### Partial Match Metrics

**Definition**: Metrics that give credit for partial entity matches.

**Types**:

- **Overlap**: Entity spans overlap but don't match exactly
- **Type Match**: Correct entity type but incorrect boundaries
- **Partial F1**: F1 that gives partial credit for partial matches

**Interpretation**:

- Higher than strict match metrics
- Useful for assessing "near miss" entity detection
- Gap between strict and partial metrics indicates boundary detection issues

**When to Use**:

- When exact boundaries are less critical
- To distinguish between boundary errors and complete misses

## Confidence Metrics

### Mean Confidence

**Definition**: The average confidence score assigned to predictions.

**Formula**: $\text{Mean Confidence} = \frac{\sum \text{Confidence Scores}}{\text{Number of Predictions}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Higher values indicate the model is more confident overall
- Should be correlated with accuracy for well-calibrated models

**When to Use**:

- To assess model calibration
- To evaluate whether confidence scores are meaningful

### Confidence-Weighted Accuracy

**Definition**: Accuracy weighted by the model's confidence in each prediction.

**Formula**: $\text{Weighted Accuracy} = \frac{\sum(\text{Correct} \cdot \text{Confidence})}{\sum \text{Confidence}}$

**Interpretation**:

- Range: 0.0 to 1.0
- Compares actual performance with the model's expectations
- A well-calibrated model should have similar standard and weighted accuracy

**When to Use**:

- To evaluate confidence calibration
- When confidence scores are used for decision making

### Confidence Calibration Error

**Definition**: Measures how well confidence scores align with actual probabilities of correctness.

**Formula**: Calculated as the mean squared error between confidence and correctness

**Interpretation**:

- Lower is better
- Values near 0 indicate well-calibrated confidence
- High values indicate poor calibration

**When to Use**:

- When using confidence scores for thresholding
- When confidence scores drive important decisions

## Aggregate Metrics

### Micro-Average

**Definition**: Calculate metrics globally by counting total true positives, false positives, and false negatives across all classes.

**Formula**: Calculate precision, recall, and F1 using combined counts across all classes

**Interpretation**:

- Gives equal weight to each example
- Influenced more by high-frequency classes
- Single value representing overall performance

**When to Use**:

- When you care about overall performance
- When class imbalance reflects the real-world distribution

### Macro-Average

**Definition**: Calculate metrics for each class independently and then average them.

**Formula**: Average of per-class precision, recall, and F1 scores

**Interpretation**:

- Gives equal weight to each class regardless of frequency
- Less influenced by dominant classes
- Better for assessing performance across imbalanced classes

**When to Use**:

- When all classes are equally important regardless of frequency
- To identify if the model struggles with particular classes

### Weighted-Average

**Definition**: Like macro-average, but weighted by the support of each class.

**Formula**: Average of per-class metrics weighted by support

**Interpretation**:

- Balance between micro and macro averaging
- Classes with more examples have more influence, but not dominance
- Often preferred for reporting overall performance

**When to Use**:

- When class importance is proportional to frequency
- For a balanced view of performance

## Custom and Advanced Metrics

### Out-of-Scope Detection Accuracy

**Definition**: Accuracy of correctly identifying inputs that don't match any defined intent.

**Interpretation**:

- Higher values indicate better detection of out-of-scope queries
- Critical for handling unexpected user inputs

**When to Use**:

- When handling out-of-scope queries is important
- In open-domain conversational systems

### Confusion Index

**Definition**: Measures how frequently the model confuses specific intent pairs.

**Formula**: Normalized count of confusions between each intent pair

**Interpretation**:

- Higher values indicate more confusion between intents
- Useful for identifying intent pairs that need better separation

**When to Use**:

- When refining intent definitions
- To identify semantically similar intents

### K-Error Reduction

**Definition**: The reduction in error rate after implementing specific improvements.

**Formula**: $\text{K-Error Reduction} = \frac{\text{Old Error Rate} - \text{New Error Rate}}{\text{Old Error Rate}}$

**Interpretation**:

- Range: 0.0 to 1.0 (or can be negative if performance degrades)
- A value of 0.3 means error rate was reduced by 30%
- Useful for measuring improvement impact

**When to Use**:

- To quantify improvements between model versions
- When reporting progress to stakeholders
