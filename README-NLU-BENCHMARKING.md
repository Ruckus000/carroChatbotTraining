# NLU Model Benchmarking System

A comprehensive framework for benchmarking, tracking, and visualizing NLU model performance with integrated CI/CD capabilities.

## Features

- **Modular Evaluation**: Efficient, optimized evaluation of intent and entity recognition
- **Metrics Tracking**: Track performance metrics over time with CSV and JSON exports
- **Interactive Visualization**: Beautiful, interactive visualizations and HTML reports
- **Model Pipeline**: Full model lifecycle management with versioning and quality gates
- **Regression Testing**: Statistical significance testing with configurable thresholds
- **CI/CD Integration**: GitHub Actions workflow for automated benchmarking
- **Streamlit Dashboard**: Interactive dashboard for exploring performance metrics

## Table of Contents

- [Quick Start](#quick-start)
- [System Components](#system-components)
- [Usage Guide](#usage-guide)
- [Dashboard](#dashboard)
- [Model Pipeline](#model-pipeline)
- [Regression Testing](#regression-testing)
- [CI/CD Integration](#cicd-integration)
- [Configuration](#configuration)

## Quick Start

1. **Installation**:

   ```bash
   # Install core requirements
   pip install -r requirements.txt

   # Install dashboard requirements (optional)
   pip install -r requirements-dashboard.txt
   ```

2. **Evaluate a model**:

   ```bash
   python evaluate_nlu.py --benchmark data/benchmark_dataset.json
   ```

3. **Launch dashboard**:

   ```bash
   ./run_dashboard.sh
   ```

4. **Run full pipeline**:
   ```bash
   python model_pipeline.py pipeline \
     --training-data data/nlu_training_data.json \
     --benchmark-data data/benchmark_dataset.json
   ```

## System Components

### Core Components

- **evaluate_nlu.py**: Core evaluation script with entity/intent metrics calculation
- **model_pipeline.py**: Model lifecycle management, versioning and quality gates
- **test_nlu_regression.py**: Regression testing with statistical significance testing
- **nlu_dashboard.py**: Streamlit dashboard for visualization and exploration

### Support Files

- **regression_config.yml**: Configuration for regression testing thresholds
- **run_dashboard.sh**: Launcher script for the Streamlit dashboard
- **.github/workflows/nlu-ci.yml**: GitHub Actions CI/CD integration

## Usage Guide

### Basic Evaluation

Evaluate an NLU model's performance against a benchmark dataset:

```bash
python evaluate_nlu.py \
  --benchmark data/benchmark_dataset.json \
  --model trained_nlu_model
```

#### Options:

- `--benchmark`: Path to benchmark data file (default: data/benchmark_dataset.json)
- `--model`: Path to the trained model (default: trained_nlu_model)
- `--output`: Output directory for results (default: benchmark_results)
- `--model-id`: Optional model identifier for tracking (default: timestamped)
- `--no-vis`: Skip generating visualizations
- `--html-report`: Generate an HTML report with visualizations
- `--verbose`: Show detailed progress information

### Visualizing Results

After evaluation, visualization files are stored in:

- JSON metrics: `benchmark_results/metrics_TIMESTAMP.json`
- CSV history: `benchmark_results/metrics_history.csv`
- Plots: `benchmark_results/visualizations/`
- HTML report: `benchmark_results/report_TIMESTAMP.html`

## Dashboard

The Streamlit dashboard provides an interactive way to explore model performance.

### Launching the Dashboard

```bash
./run_dashboard.sh
```

The dashboard automatically runs on http://localhost:8501

### Dashboard Features

- **Model Comparison**: Compare multiple model versions side by side
- **Intent Analysis**: Detailed per-intent performance metrics and confusion matrix
- **Entity Analysis**: Entity extraction performance across entity types
- **Error Analysis**: Examine specific error cases and patterns
- **Performance Trends**: Track metrics over time with interactive charts

## Model Pipeline

The model pipeline provides comprehensive model lifecycle management.

### Commands

```bash
# Train a new model
python model_pipeline.py train --data data/nlu_training_data.json

# Benchmark a model
python model_pipeline.py benchmark --model-id model_20230615_123456

# Run regression tests
python model_pipeline.py regression --model-id model_20230615_123456

# List all models
python model_pipeline.py list

# Export a model for deployment
python model_pipeline.py export --model-id model_20230615_123456

# Run full pipeline (train, benchmark, regression test)
python model_pipeline.py pipeline \
  --training-data data/nlu_training_data.json \
  --benchmark-data data/benchmark_dataset.json
```

### Model Registry

The pipeline maintains a model registry in `models/model_registry.json` with:

- Model metadata
- Training information
- Benchmark results
- Regression test results

## Regression Testing

The regression testing system detects performance degradation by:

1. Comparing metrics against historical values
2. Applying statistical significance tests
3. Checking configurable thresholds (absolute and relative)
4. Generating visual trend analysis

```bash
# Run regression test with default config
python test_nlu_regression.py --metrics-file benchmark_results/metrics_20230615_123456.json

# With custom config
python test_nlu_regression.py --config regression_config.yml --ci
```

## CI/CD Integration

GitHub Actions workflow automatically:

1. Runs on code or data changes
2. Executes regression tests on the current model
3. Optionally trains and benchmarks new models
4. Reports results as artifacts

To trigger model training from a commit, include `[train]` in the commit message.

## Configuration

### Regression Testing Configuration

The `regression_config.yml` file controls:

- Thresholds for metrics (absolute, relative, minimum values)
- Class-specific thresholds for critical intents/entities
- Statistical testing parameters
- CI behavior and reporting
- Visualization settings

Example threshold configuration:

```yaml
thresholds:
  intent:
    accuracy:
      absolute: 0.05 # Max 5 percentage point drop
      relative: 5.0 # Max 5% relative drop
      min_value: 0.75 # Never accept below 75% accuracy
```

## Advanced Topics

### Adding Custom Metrics

To add custom metrics:

1. Add metric calculation in the `calculate_metrics()` function in `evaluate_nlu.py`
2. Update visualization in `create_visualizations()`
3. Add thresholds in `regression_config.yml`

### Integration with Training Pipeline

The benchmarking system can be integrated with your training pipeline by:

1. Calling `model_pipeline.py` from your training scripts
2. Using the `--model-id` parameter to track specific model versions
3. Configuring quality gates with the regression testing system

## Troubleshooting

### Common Issues

- **Missing dependencies**: Ensure all requirements are installed from both requirements files
- **Visualization errors**: Verify matplotlib and seaborn are properly installed
- **Dashboard not starting**: Check that Streamlit is installed correctly

### Logs

- Benchmark logs: Standard output and in metrics JSON files
- Dashboard logs: In terminal where dashboard is launched
- Pipeline logs: Standard output with INFO level details
