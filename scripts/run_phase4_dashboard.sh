#!/bin/bash

# Run Phase 4 NLU Dashboard
# - With enhanced UI and UX components

# Set script directory as working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check for virtual environment
if [ ! -d "venv_dashboard" ]; then
    echo "Creating virtual environment for dashboard..."
    python3 -m venv venv_dashboard
fi

# Activate virtual environment
source venv_dashboard/bin/activate

# Check if requirements are installed
pip install -q -r config/requirements-dashboard.txt

# Create the benchmark results directory if it doesn't exist
if [ ! -d "benchmark_results" ]; then
    echo "Creating benchmark results directory..."
    mkdir -p benchmark_results
fi

# Create the CSS directory if it doesn't exist
if [ ! -d "assets/css" ]; then
    echo "Creating CSS assets directory..."
    mkdir -p assets/css
fi

# Run the Streamlit app
echo "Starting NLU Dashboard with Phase 4 enhancements..."
echo "Open your browser at http://localhost:8501 to view the dashboard"
streamlit run src/nlu_dashboard.py

# Deactivate virtual environment
deactivate 