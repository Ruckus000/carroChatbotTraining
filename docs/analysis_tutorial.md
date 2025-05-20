# NLU Analysis Tutorial

This step-by-step tutorial guides you through common analysis scenarios using the NLU Benchmarking Dashboard. Each tutorial provides a concrete workflow for addressing specific challenges in NLU model development.

## Table of Contents

1. [Identifying and Fixing Confused Intents](#identifying-and-fixing-confused-intents)
2. [Improving Entity Recognition Performance](#improving-entity-recognition-performance)
3. [Analyzing Cross-Intent Entities](#analyzing-cross-intent-entities)
4. [Diagnosing Confidence Issues](#diagnosing-confidence-issues)
5. [Tracking Performance Changes Across Versions](#tracking-performance-changes-across-versions)

## Identifying and Fixing Confused Intents

### Scenario

Your chatbot frequently confuses similar intents, leading to incorrect responses.

### Step-by-Step Analysis

1. **Identify Confused Intent Pairs**

   - Navigate to **Latest Results > Intent Performance**
   - Study the confusion matrix
   - Look for off-diagonal cells with high values (bright spots)
   - Note: Larger values indicate more frequent confusions

2. **Analyze Sample Utterances**

   - Navigate to **Error Explorer**
   - Filter for specific intent pairs (e.g., From "book_flight" To "book_hotel")
   - Click on "Show Examples" to see the confused utterances
   - Analyze patterns in the misclassified examples

3. **Review Intent Definitions**

   - Are the intents semantically similar?
   - Do they share significant vocabulary?
   - Is one intent a subset of another?
   - Are there clear distinguishing features?

4. **Implement Solutions**

   - **Option A: Refine Intent Definitions**
     - Make intent definitions more distinct
     - Update training examples to emphasize differences
   - **Option B: Merge Similar Intents**
     - If intents are genuinely similar, consider combining them
     - Use entities to differentiate specific subtypes
   - **Option C: Add More Training Data**
     - Add examples that highlight the differences between intents
     - Focus on boundary cases that clarify the distinction

5. **Test and Iterate**
   - Retrain the model with the updated data
   - Run a new benchmark
   - Check if the confusion has been reduced
   - Repeat if necessary

### Example Case

**Problem**: Confusion between "set_alarm" and "set_reminder" intents.

**Analysis**:

- Confusion matrix shows 23% of "set_reminder" utterances classified as "set_alarm"
- Examples revealed phrases like "remind me at 7am" being classified as "set_alarm"
- Both intents share time-related vocabulary but have different purposes

**Solution**:

- Added 15 examples emphasizing the notification aspect of reminders
- Added 10 examples highlighting the repeating nature of alarms
- Updated intent guidelines to clarify the distinction

**Result**:

- Confusion dropped from 23% to 7%
- Overall intent accuracy improved by 4.3%

## Improving Entity Recognition Performance

### Scenario

Your model correctly classifies intents but struggles with extracting certain entity types accurately.

### Step-by-Step Analysis

1. **Identify Problematic Entities**

   - Navigate to **Latest Results > Entity Performance**
   - Look for entities with low F1 scores
   - Pay attention to the precision/recall breakdown
   - Check the support column to ensure adequate sample size

2. **Understand the Error Type**

   - Navigate to **Error Explorer > Entity Errors**
   - Filter by the problematic entity type
   - Determine if the issue is:
     - **Precision issue**: Many false positives (wrong entity detections)
     - **Recall issue**: Many false negatives (missed entity detections)
     - **Boundary issue**: Entities detected but with incorrect spans

3. **Analyze Examples**

   - For boundary issues:
     - Compare correct and predicted spans
     - Look for patterns in the differences
   - For precision issues:
     - Identify what's being incorrectly tagged
     - Look for patterns that might confuse the model
   - For recall issues:
     - Examine missed examples
     - Check for unusual formatting or contexts

4. **Implement Solutions**

   - **For Boundary Issues**:
     - Standardize annotation guidelines
     - Add examples with clear entity boundaries
     - Consider using regex for structured entities
   - **For Precision Issues**:
     - Add negative examples (similar text without the entity)
     - Clarify entity definition guidelines
   - **For Recall Issues**:
     - Add more diverse examples of the entity
     - Include examples in varied contexts
     - Add examples of rare variants of the entity

5. **Test and Iterate**
   - Retrain the model with updated data
   - Run a new benchmark
   - Verify improvement in entity metrics
   - Continue refining until performance is satisfactory

### Example Case

**Problem**: Low recall (0.63) for "duration" entity.

**Analysis**:

- Most missed examples used uncommon time formats
- Detected examples mostly used standard formats like "X minutes" or "X hours"
- Missed examples often used fractions or ranges ("2.5 hours", "3-4 days")

**Solution**:

- Added 20 examples with varied time formats
- Included 5 examples with fractional durations
- Added 8 examples with time ranges

**Result**:

- Entity recall improved from 0.63 to 0.82
- Overall entity F1 score increased by 7.2%

## Analyzing Cross-Intent Entities

### Scenario

Entities that appear across multiple intents show inconsistent performance.

### Step-by-Step Analysis

1. **Identify Cross-Intent Entities**

   - Navigate to **Guided Analysis > Entity Distribution**
   - Identify entities that appear in multiple intents
   - Note intents where the entity performs well vs. poorly

2. **Compare Performance Across Intents**

   - Click on the entity name to see per-intent performance
   - Look for significant performance differences
   - Check sample counts to ensure adequate representation

3. **Analyze Contextual Differences**

   - Navigate to **Error Explorer > Entity Errors**
   - Filter by entity type and compare across intents
   - Look for:
     - Different linguistic contexts
     - Different entity formats
     - Different co-occurring entities

4. **Implement Solutions**

   - **Balance training examples**:
     - Ensure adequate examples of the entity across all intents
   - **Standardize entity formats**:
     - Use consistent annotation patterns across intents
   - **Add challenging examples**:
     - For high-performing intents, add more challenging cases
     - For low-performing intents, add more diverse examples

5. **Test and Iterate**
   - Retrain and benchmark
   - Check for performance improvement
   - Look for reduced performance gaps across intents

### Example Case

**Problem**: "date" entity shows 0.92 F1 in "book_flight" but only 0.74 F1 in "check_weather".

**Analysis**:

- In "book_flight", dates typically follow standard formats (e.g., "January 15th")
- In "check_weather", dates often use relative expressions ("next week", "this weekend")
- "book_flight" had 3x more training examples with dates than "check_weather"

**Solution**:

- Added 18 examples to "check_weather" with relative date expressions
- Added 10 examples to "book_flight" with more relative date formats
- Standardized annotation guidelines for relative dates

**Result**:

- "date" F1 in "check_weather" improved to 0.86
- Consistency between intents improved by 57%

## Diagnosing Confidence Issues

### Scenario

Your model gives incorrect predictions with high confidence or correct predictions with low confidence.

### Step-by-Step Analysis

1. **Assess Confidence Distribution**

   - Navigate to **Latest Results > Confidence Analysis**
   - Compare confidence distributions for correct vs. incorrect predictions
   - Look for:
     - High-confidence incorrect predictions
     - Low-confidence correct predictions
     - Overall calibration curve

2. **Identify Problematic Patterns**

   - Navigate to **Error Explorer**
   - Filter for high-confidence errors (e.g., confidence > 0.8)
   - Look for patterns in these examples
   - Repeat for low-confidence correct predictions

3. **Analyze by Intent and Entity Type**

   - Check if confidence issues affect specific intents/entities
   - Look for intents with wide confidence variation
   - Identify entities with consistently low confidence

4. **Implement Solutions**

   - **For high-confidence errors**:
     - Add more examples of commonly confused cases
     - Review and possibly merge very similar intents
     - Consider more complex model architecture
   - **For low-confidence correct predictions**:
     - Add more varied examples of these cases
     - Consider adjusting confidence thresholds for specific intents
     - Evaluate if these examples are genuinely ambiguous

5. **Test and Iterate**
   - Retrain the model with updated data
   - Run a new benchmark with confidence analysis
   - Check if the confidence calibration has improved

### Example Case

**Problem**: "request_weather" intent has high accuracy (0.95) but low average confidence (0.71).

**Analysis**:

- Most low-confidence but correct examples contained multiple locations or time periods
- Model correctly classified these but with uncertainty
- Examples were more complex than typical training data

**Solution**:

- Added 25 examples with multiple locations or time periods
- Added 10 examples with more complex structures
- Used data augmentation to create variations of existing examples

**Result**:

- Average confidence for correct predictions increased to 0.88
- Maintained high accuracy (0.96)
- Reduced confidence calibration error by 42%

## Tracking Performance Changes Across Versions

### Scenario

You need to evaluate how model changes impact performance over time and identify regressions.

### Step-by-Step Analysis

1. **Compare Overall Metrics**

   - Navigate to **Performance History**
   - Check the trend lines for key metrics
   - Identify points of significant change
   - Note version changes that correlate with metric changes

2. **Drill Down to Intent Level**

   - Click on specific data points to see version details
   - Use the **Model Comparison** tool to compare versions
   - Identify intents with the largest performance changes

3. **Analyze Entity Performance Changes**

   - Switch to entity metrics in **Performance History**
   - Identify entities with significant changes
   - Correlate with model or data changes

4. **Identify Regression Causes**

   - For negative changes, use **Model Comparison > Regression Analysis**
   - Compare example predictions between versions
   - Look for patterns in newly misclassified examples

5. **Implement Solutions**
   - **For regressions**:
     - Add back important examples that may have been lost
     - Consider rolling back problematic changes
     - Create targeted examples for affected intents/entities
   - **For improvements**:
     - Document successful strategies
     - Apply similar approaches to other areas
     - Consider further enhancing successful approaches

### Example Case

**Problem**: After v2.3 update, overall intent F1 improved by 3%, but "process_payment" intent F1 dropped by 8%.

**Analysis**:

- Model comparison showed 12 new errors in "process_payment"
- Most new errors were now classified as "check_balance"
- Changes were traced to new payment-related examples added to "check_balance"

**Solution**:

- Added 15 targeted examples to distinguish payment processing from balance checking
- Created clearer guidelines for annotating these intents
- Added specific examples for boundary cases between the two intents

**Result**:

- "process_payment" F1 recovered to previous level plus 2% improvement
- Maintained improvements in other intents
- Reduced overall regression risk in future updates
