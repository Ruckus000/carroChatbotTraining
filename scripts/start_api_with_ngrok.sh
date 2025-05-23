#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}NLU Chatbot API with ngrok Tunnel${NC}"
echo "----------------------------------------"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo -e "${RED}Error: ngrok is not installed. Please run: npm install -g ngrok${NC}"
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
pip install -r config/requirements-api.txt > /dev/null 2>&1
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

# Kill any existing processes
echo -e "${YELLOW}Stopping any existing servers...${NC}"
pkill -f "python.*api.py" || true
pkill -f "ngrok.*http.*8001" || true
sleep 2

# Show startup instructions
echo -e "\n\033[1;36m=======================================\033[0m"
echo -e "\033[1;36m🚀 Starting API Server with ngrok 🚀\033[0m"
echo -e "\033[1;36m=======================================\033[0m\n"

echo -e "Starting services in the following order:"
echo -e "1. ${CYAN}API Server${NC} on localhost:8001"
echo -e "2. ${PURPLE}ngrok tunnel${NC} to expose the API publicly"
echo -e "\n${GREEN}Please wait while services start...${NC}\n"

# Start the API server in the background
echo -e "${CYAN}[1/2] Starting API Server...${NC}"
export PYTHONPATH="$(pwd):$PYTHONPATH"
cd src && PORT=8001 python api.py > ../api_server.log 2>&1 &
API_PID=$!
cd ..

# Wait a moment for the server to start
sleep 5

# Check if API server is running
if ! curl -s http://localhost:8001/api/health > /dev/null; then
    echo -e "${RED}❌ API Server failed to start. Check api_server.log for details.${NC}"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ API Server started successfully${NC}"

# Start ngrok tunnel
echo -e "${PURPLE}[2/2] Starting ngrok tunnel...${NC}"
/opt/homebrew/bin/ngrok http 8001 > ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to establish tunnel
sleep 3

# Get ngrok URL
NGROK_URL=""
for i in {1..10}; do
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for tunnel in data['tunnels']:
        if tunnel['proto'] == 'https':
            print(tunnel['public_url'])
            break
except:
    pass
" 2>/dev/null)
    
    if [ ! -z "$NGROK_URL" ]; then
        break
    fi
    echo -e "${YELLOW}Waiting for ngrok tunnel... (${i}/10)${NC}"
    sleep 2
done

if [ -z "$NGROK_URL" ]; then
    echo -e "${RED}❌ Failed to get ngrok URL. Check ngrok.log for details.${NC}"
    kill $API_PID $NGROK_PID 2>/dev/null
    exit 1
fi

# Display connection information
echo -e "\n\033[1;32m🎉 SUCCESS! Both services are running! 🎉\033[0m\n"

echo -e "\033[1;36m📱 CONNECTION INFORMATION 📱\033[0m"
echo -e "=================================="
echo -e "\033[1;33mPublic ngrok URL (recommended):\033[0m"
echo -e "${NGROK_URL}/api"
echo -e "\n\033[1;33mLocal development URLs:\033[0m"
echo -e "• iOS Simulator: http://localhost:8001/api"
echo -e "• Android Emulator: http://10.0.2.2:8001/api"
echo -e "\n\033[1;33mMonitoring:\033[0m"
echo -e "• ngrok Web Interface: http://localhost:4040"
echo -e "• API Logs: tail -f api_server.log"
echo -e "• ngrok Logs: tail -f ngrok.log"

echo -e "\n\033[1;36m📋 FOR YOUR REACT NATIVE APP 📋\033[0m"
echo -e "=================================="
echo -e "Update your \033[1;33menv.js\033[0m file with:"
echo -e "\033[1;32mconst apiUrl = '${NGROK_URL}/api'\033[0m"

echo -e "\n\033[1;36m🔄 IMPORTANT NOTES 🔄\033[0m"
echo -e "=================================="
echo -e "• This ngrok URL will work from ANY network/WiFi"
echo -e "• The URL changes each time you restart this script"
echo -e "• For a permanent URL, sign up at ngrok.com"
echo -e "• Press \033[1;31mCtrl+C\033[0m to stop both services"
echo -e "\n\033[1;32m🟢 Services are ready! Start testing your React Native app! 🟢\033[0m\n"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    kill $API_PID $NGROK_PID 2>/dev/null
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait for user to stop
wait $API_PID $NGROK_PID 