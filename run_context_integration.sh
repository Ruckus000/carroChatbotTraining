#!/bin/bash

# Context Integration Pipeline Script
# This script runs the complete context integration pipeline:
# 1. Train context-aware models
# 2. Run unit tests
# 3. Start the hybrid Streamlit app

# Exit on error
set -e

# Print execution steps
echo "==========================================="
echo "CONTEXT INTEGRATION PIPELINE"
echo "==========================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if requirements are installed
if [ ! -f "requirements_checked.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch requirements_checked.txt
fi

# Create output directories
echo "Creating output directories..."
mkdir -p output/models/context_aware
mkdir -p output/data/context_aware

# Train context-aware models
echo "==========================================="
echo "TRAINING CONTEXT-AWARE MODELS"
echo "==========================================="
python train_context_models.py

# Run unit tests
echo "==========================================="
echo "RUNNING UNIT TESTS"
echo "==========================================="
python test_context_integration.py

# Check test result
if [ $? -eq 0 ]; then
    echo "Tests passed successfully!"
else
    echo "Tests failed! Please check the errors above."
    exit 1
fi

# Start Streamlit app
echo "==========================================="
echo "STARTING STREAMLIT APP"
echo "==========================================="
echo "The Streamlit app will start now with both standard and context-aware modes available."
echo "Press Ctrl+C to stop the app when done."
streamlit run streamlit_app.py

# End message
echo "==========================================="
echo "CONTEXT INTEGRATION PIPELINE COMPLETED"
echo "===========================================" 