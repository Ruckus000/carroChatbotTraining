#!/bin/bash

# Run script for NLU Benchmarking Dashboard with Phase 5 features
# This script sets up the environment and launches the dashboard

# Text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Bold text
BOLD='\033[1m'
NORMAL='\033[0m'

echo -e "${BLUE}${BOLD}NLU Benchmarking Dashboard${NORMAL}${NC} - ${GREEN}Phase 5: Documentation and Educational Elements${NC}"
echo -e "${YELLOW}Setting up environment and launching dashboard...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3.7+ and try again.${NC}"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 7 ]); then
    echo -e "${RED}Error: Python 3.7+ is required. Found Python $python_version${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv_dashboard" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv_dashboard
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install venv package and try again.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv_dashboard/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi

# Install or upgrade required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install --upgrade pip > /dev/null
pip install -r requirements-dashboard.txt > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install required packages. See error above.${NC}"
    exit 1
fi

# Check if benchmark results directory exists
if [ ! -d "benchmark_results" ]; then
    echo -e "${YELLOW}Creating benchmark results directory...${NC}"
    mkdir -p benchmark_results
fi

# Check if educational materials are available
echo -e "${YELLOW}Checking documentation and educational materials...${NC}"
docs_files=("docs/dashboard_user_guide.md" "docs/metrics_glossary.md" "docs/analysis_tutorial.md" "docs/troubleshooting_guide.md" "docs/nlu_best_practices.md")
all_docs_exist=true

for doc_file in "${docs_files[@]}"; do
    if [ ! -f "$doc_file" ]; then
        echo -e "${RED}Warning: $doc_file not found. Some educational features may not be available.${NC}"
        all_docs_exist=false
    fi
done

if $all_docs_exist; then
    echo -e "${GREEN}All educational materials are available.${NC}"
fi

# Set environment variables for Phase 5 features
export NLU_DASHBOARD_PHASE=5
export NLU_DASHBOARD_DOCS_PATH="./docs"
export NLU_DASHBOARD_ENABLE_EDUCATIONAL=true

# Show startup message
echo -e "\n${GREEN}${BOLD}Starting NLU Benchmarking Dashboard (Phase 5)${NORMAL}${NC}"
echo -e "${BLUE}Features enabled:${NC}"
echo -e "  • ${YELLOW}Enhanced UI and Visualizations${NC}"
echo -e "  • ${YELLOW}Interactive Analysis Tools${NC}"
echo -e "  • ${YELLOW}Help Content and Tooltips${NC}"
echo -e "  • ${YELLOW}Educational Materials and Documentation${NC}"
echo -e "  • ${YELLOW}Comprehensive User Guide${NC}"
echo -e "\n${YELLOW}The dashboard will open in your web browser momentarily...${NC}\n"

# Start the Streamlit app
streamlit run src/nlu_dashboard.py

# Deactivate virtual environment when done
deactivate 