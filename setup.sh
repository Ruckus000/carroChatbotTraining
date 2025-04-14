#!/bin/bash

# Make the training script executable
chmod +x run_training.sh

# Set up Python virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
    
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Install requirements
    echo "Installing required packages..."
    pip install -r requirements.txt
    
    echo "Environment setup complete!"
else
    echo "Virtual environment already exists. Activating..."
    source .venv/bin/activate
    
    echo "You may want to update packages:"
    echo "pip install -r requirements.txt"
fi

echo ""
echo "To run the chatbot training, execute:"
echo "./run_training.sh"
