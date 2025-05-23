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
pip install -r config/requirements-api.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies${NC}"
    exit 1
fi

# Check if trained_nlu_model exists
if [ ! -d "trained_nlu_model" ]; then
    echo -e "${RED}Error: trained_nlu_model directory not found${NC}"
    echo "Please run 'python src/train.py' to train the NLU model first"
    exit 1
fi

# Kill any existing API server processes
pkill -f "python src/api.py" || true

# Show startup instructions
echo -e "\n\033[1;36m=======================================\033[0m"
echo -e "\033[1;36m📱 Starting API Server 📱\033[0m"
echo -e "\033[1;36m=======================================\033[0m\n"

echo -e "API server will start on port 8001"
echo -e "You will see highlighted chat messages in this terminal"

# Get all IP addresses
echo -e "\n\033[1;33m🌐 AVAILABLE CONNECTION OPTIONS 🌐\033[0m"
echo -e "====================================="

# Local development
echo -e "\033[1;32m📱 For iOS Simulator:\033[0m"
echo -e "http://localhost:8001/api"
echo -e "http://127.0.0.1:8001/api"

echo -e "\n\033[1;32m🤖 For Android Emulator:\033[0m"
echo -e "http://10.0.2.2:8001/api"

# Physical devices on same WiFi
echo -e "\n\033[1;32m📲 For Physical Devices (Same WiFi):\033[0m"
# Get local IP addresses
LOCAL_IPS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -3)
if [ ! -z "$LOCAL_IPS" ]; then
    while IFS= read -r ip; do
        echo -e "http://${ip}:8001/api"
    done <<< "$LOCAL_IPS"
else
    echo -e "Unable to detect local IP addresses"
fi

echo -e "\n\033[1;32m🌍 For Any Network (ngrok required):\033[0m"
echo -e "Run: ./scripts/start_api_with_ngrok.sh"
echo -e "See: docs/NGROK_SETUP_GUIDE.md for setup instructions"

echo -e "\n\033[1;36m💡 QUICK SETUP TIPS 💡\033[0m"
echo -e "======================="
echo -e "• Use localhost for iOS Simulator"
echo -e "• Use 10.0.2.2 for Android Emulator"  
echo -e "• Use your machine's IP for physical devices"
echo -e "• Use ngrok for testing across different networks"

echo -e "\n\033[1;35m🔧 React Native Configuration Examples:\033[0m"
echo -e "========================================="
echo -e "// iOS Simulator"
echo -e "const apiUrl = 'http://localhost:8001/api';"
echo -e ""
echo -e "// Android Emulator"  
echo -e "const apiUrl = 'http://10.0.2.2:8001/api';"
echo -e ""
echo -e "// Physical Device (replace with your IP)"
if [ ! -z "$LOCAL_IPS" ]; then
    FIRST_IP=$(echo "$LOCAL_IPS" | head -1)
    echo -e "const apiUrl = 'http://${FIRST_IP}:8001/api';"
fi

echo -e "\n\033[1;32m🟢 Starting API server...\033[0m"
echo -e "Press \033[1;31mCtrl+C\033[0m to stop the server\n"

# Start the API server
export PYTHONPATH="$(pwd):$PYTHONPATH"
cd src && PORT=8001 python api.py 