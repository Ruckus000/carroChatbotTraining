#!/bin/bash

# Configuration
DASHBOARD_SCRIPT="nlu_dashboard.py"
REQUIREMENTS_FILE="requirements-dashboard.txt"
STREAMLIT_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print styled message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_message "$RED" "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv_dashboard" ]; then
    print_message "$YELLOW" "Creating virtual environment for dashboard..."
    python3 -m venv venv_dashboard
    if [ $? -ne 0 ]; then
        print_message "$RED" "Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then # Windows
    source venv_dashboard/Scripts/activate
else # Linux/macOS
    source venv_dashboard/bin/activate
fi

if [ $? -ne 0 ]; then
    print_message "$RED" "Failed to activate virtual environment."
    exit 1
fi

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    print_message "$YELLOW" "Installing required packages..."
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        print_message "$RED" "Failed to install required packages."
        exit 1
    fi
fi

# Check if benchmark results directory exists
if [ ! -d "benchmark_results" ]; then
    print_message "$YELLOW" "Warning: No benchmark results found. Run evaluation first."
    mkdir -p benchmark_results
fi

# Start the dashboard
print_message "$GREEN" "Starting NLU Benchmarking Dashboard on port $STREAMLIT_PORT..."
streamlit run "$DASHBOARD_SCRIPT" -- --server.port=$STREAMLIT_PORT --server.headless=$STREAMLIT_SERVER_HEADLESS

# Deactivate virtual environment on exit
deactivate 