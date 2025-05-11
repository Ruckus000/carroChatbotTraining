#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}NLU Chatbot API Starter Script${NC}"
echo "---------------------------------"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements-api.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies${NC}"
    exit 1
fi

# Check if trained_nlu_model exists
if [ ! -d "trained_nlu_model" ]; then
    echo -e "${RED}Error: trained_nlu_model directory not found${NC}"
    echo "Please run 'python train.py' to train the NLU model first"
    exit 1
fi

# Kill any existing API server processes
pkill -f "python api.py" || true

# Show startup instructions
echo -e "\n\033[1;36m============================\033[0m"
echo -e "\033[1;36m📱 Starting API Server 📱\033[0m"
echo -e "\033[1;36m============================\033[0m\n"
echo -e "API server will start on port 8001"
echo -e "You will see highlighted chat messages in this terminal"
echo -e "Connect your React Native app to: http://YOUR_MACHINE_IP:8001\n"
echo -e "\033[1;32mAPI server starting...\033[0m\n"

# Start the API server
PORT=8001 python api.py 