# NLU Dashboard User Guide

This comprehensive guide explains how to use the NLU Benchmarking Dashboard to analyze, interpret, and leverage your model's performance metrics to drive improvements.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Layout and Navigation](#dashboard-layout-and-navigation)
3. [Understanding Key Metrics](#understanding-key-metrics)
4. [Analyzing Intent Performance](#analyzing-intent-performance)
5. [Analyzing Entity Recognition](#analyzing-entity-recognition)
6. [Error Analysis Techniques](#error-analysis-techniques)
7. [Historical Performance Tracking](#historical-performance-tracking)
8. [Model Comparison](#model-comparison)
9. [Guided Analysis](#guided-analysis)
10. [Exporting and Sharing Results](#exporting-and-sharing-results)
11. [Troubleshooting](#troubleshooting)
12. [Common Improvement Strategies](#common-improvement-strategies)

## Getting Started

### Installation Requirements

- Python 3.7+
- Streamlit 1.22.0+
- Required packages listed in `requirements-dashboard.txt`

### Running the Dashboard

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements-dashboard.txt
   ```
3. Run the dashboard:
   ```bash
   ./run_phase4_dashboard.sh
   ```
   Or directly with:
   ```bash
   streamlit run nlu_dashboard.py
   ```
4. The dashboard will open in your default web browser at `http://localhost:8501`

## Dashboard Layout and Navigation

### Navigation Menu

The left sidebar contains the main navigation menu with these sections:

- **Home**: Overview and quick access to key functions
- **Latest Results**: Detailed analysis of the most recent benchmark run
- **Performance History**: Track metrics over time across model versions
- **Error Analysis**: Deep dive into model errors and patterns
- **Model Comparison**: Side-by-side analysis of different model versions
- **Guided Analysis**: Step-by-step analysis workflow for model improvement

### Interactive Dashboard Tour

Use the "Dashboard Tour" button in the sidebar to start an interactive tour of all features. The tour walks through each section with explanations of what you're seeing and how to interpret it.

## Understanding Key Metrics

### Intent Classification Metrics

- **Accuracy**: Percentage of examples where the predicted intent matches the true intent

  - _Interpretation_: Higher is better. A score of 0.95 means the model correctly predicted the intent for 95% of examples
  - _Limitations_: Can be misleading with imbalanced datasets

- **Precision**: How many of the predicted instances of an intent are actually correct

  - _Formula_: True Positives / (True Positives + False Positives)
  - _Interpretation_: Higher precision means fewer false positives

- **Recall**: How many of the actual instances of an intent are correctly predicted

  - _Formula_: True Positives / (True Positives + False Negatives)
  - _Interpretation_: Higher recall means fewer false negatives

- **F1 Score**: Harmonic mean of precision and recall
  - _Formula_: 2 _ (Precision _ Recall) / (Precision + Recall)
  - _Interpretation_: Balances precision and recall; best metric for imbalanced datasets

### Entity Recognition Metrics

- **Entity Precision**: How many of the predicted entities are correct
- **Entity Recall**: How many of the actual entities were correctly predicted
- **Entity F1**: Harmonic mean of entity precision and recall
- **Support**: Number of examples for each entity type

### Color-Coded Performance Indicators

Performance indicators in the dashboard use color-coding for quick assessment:

- **Green** (>0.9): Excellent performance
- **Light Green** (0.8-0.9): Good performance
- **Orange** (0.7-0.8): Fair performance, consider improvements
- **Red** (<0.7): Poor performance, needs attention

## Analyzing Intent Performance

### Confusion Matrix

The confusion matrix shows patterns of correct and incorrect predictions:

- **Rows**: True intents (what the intent actually is)
- **Columns**: Predicted intents (what the model predicted)
- **Diagonal**: Correct predictions (true intent = predicted intent)
- **Off-diagonal**: Incorrect predictions (true intent ≠ predicted intent)

#### How to Use the Confusion Matrix

1. Look for bright spots off the diagonal - these indicate common misclassifications
2. Check for clustered errors - these suggest semantically similar intents
3. Use the expanded view to see specific counts and percentages

### Intent Performance Radar Chart

The radar chart shows performance for the best and worst performing intents:

- Outer points indicate better performance (F1 score closer to 1)
- Inner points indicate worse performance (F1 score closer to 0)

#### How to Analyze the Radar Chart

1. Identify the lowest-scoring intents for targeted improvements
2. Compare patterns between high and low-performing intents
3. Look for imbalances across different intent types

## Analyzing Entity Recognition

### Entity Metrics Breakdown

The entity recognition section provides detailed metrics for each entity type:

- **Per-entity metrics**: Precision, recall, and F1 score for each entity type
- **Micro-average**: Performance across all entity extractions equally weighted
- **Macro-average**: Average of per-entity metrics (gives equal weight to each entity)

### Entity Error Patterns

Common entity extraction errors include:

1. **Boundary errors**: Partially correct extraction (overlap but not exact match)
2. **Type errors**: Correct text span but wrong entity type
3. **Missing entities**: Failed to identify an entity (false negatives)
4. **Spurious entities**: Incorrectly identified entities (false positives)

## Error Analysis Techniques

### Error Explorer

The Error Explorer allows filtering and investigating specific error patterns:

1. Use the filter panel to focus on:

   - Specific error patterns (e.g., intent A → intent B)
   - Confidence ranges (e.g., high-confidence errors)
   - Entity types present in errors

2. Examine error examples to identify:
   - Ambiguous or unclear training examples
   - Consistently misclassified phrases or patterns
   - Entity extraction issues

### Error Pattern Analysis

The Error Pattern visualization shows the most common error types:

- **Sankey diagram**: Flow from true intents to predicted intents
- **Bar charts**: Most common and highest confidence errors
- **Confidence distribution**: Compare confidence of correct vs. incorrect predictions

### Interpreting Error Patterns

- **High confidence errors**: The model is "confidently wrong" - these need special attention
- **Clustered errors**: When multiple examples of intent A are predicted as intent B, this suggests semantic similarity
- **Low confidence correct predictions**: The model is "unsure but correct" - might need more examples to increase confidence

## Historical Performance Tracking

### Timeline Charts

The Performance History page tracks metrics across benchmark runs:

- **Line charts**: Track metrics over time to identify trends
- **Annotations**: Significant changes are highlighted
- **Model version history**: Table of all benchmarked versions

### How to Interpret Trends

1. Look for consistent improvement or degradation over time
2. Check for correlation between intent and entity performance
3. Identify sudden changes that might indicate data or model issues
4. Compare performance changes with model or data modifications

## Model Comparison

### Side-by-Side Analysis

The Model Comparison feature allows detailed comparison between two model versions:

1. Select base and comparison models from the dropdowns
2. View side-by-side metrics with highlighted differences
3. Examine most improved and degraded intents
4. Identify specific changes in performance patterns

### Effective Comparison Strategies

- Compare before/after adding new training data
- Evaluate impact of hyperparameter changes
- Assess performance differences across model architectures
- Track progress against baseline models

## Guided Analysis

The Guided Analysis workflow provides a structured approach to model analysis:

1. **Performance Overview**: Start with high-level metrics
2. **Error Analysis**: Identify and categorize common errors
3. **Improvement Recommendations**: Get actionable suggestions
4. **Implementation Planning**: Prioritize improvements based on impact

## Exporting and Sharing Results

### Export Formats

The dashboard supports exporting results in multiple formats:

- **CSV**: Tabular data for spreadsheet analysis
- **JSON**: Complete metrics data for programmatic use
- **Visualizations**: PNG images of charts and plots
- **ZIP Archive**: Bundle of all visualizations

### Sharing Results

Use the Email Results function to share findings with team members:

1. Click "Email Results" in the export section
2. Enter recipient email addresses (comma-separated)
3. Customize the subject and message
4. Select which attachments to include

## Troubleshooting

### Common Issues

- **No data appears**: Ensure benchmark runs have been completed and results saved to the expected location
- **Missing metrics**: Check that the evaluation was complete and included all necessary metrics
- **Slow loading**: Large datasets may take time to process; refresh only when needed
- **Visualization errors**: Ensure required packages (Plotly, Matplotlib) are installed correctly

### Quick Solutions

- Use the Refresh button on the home page to reload the latest data
- Check the console for any error messages if visualizations fail to load
- Verify that benchmark files exist in the expected directory

## Common Improvement Strategies

### Intent Classification Improvements

1. **Add more training examples** for low-performing intents
2. **Review and refine intent definitions** for frequently confused intents
3. **Consider merging similar intents** that are consistently confused
4. **Create distinction for ambiguous examples** with clearer intent guidelines

### Entity Recognition Improvements

1. **Standardize entity annotation guidelines** to reduce boundary errors
2. **Add examples with challenging entity contexts**
3. **Review entity type definitions** for commonly confused entities
4. **Add examples for rare entity types** to improve recall

### General Optimization Techniques

1. **Data augmentation**: Create variations of existing training examples
2. **Error-focused training**: Add more examples similar to common errors
3. **Balanced datasets**: Ensure sufficient examples across all intents and entities
4. **Cross-validation**: Use multiple validation splits to ensure consistent performance
