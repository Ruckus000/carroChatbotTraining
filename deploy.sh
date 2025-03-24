#!/bin/bash
set -e

# LangGraph Mistral Integration Deployment Script
# This script sets up and deploys the chatbot application

# Configuration
ENVIRONMENT=${1:-production}
CONFIG_DIR="./config"
CONFIG_FILE="${CONFIG_DIR}/deployment.${ENVIRONMENT}.yaml"
VENV_DIR=".venv"
LOG_DIR="./logs"
STREAMLIT_PORT=8501

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 could not be found. Please install it."
        exit 1
    fi
}

# Validate environment
check_command python3
check_command pip

# Check if environment is valid
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production|test)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    log_info "Usage: $0 [environment]"
    log_info "Valid environments: development, staging, production, test"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_warn "Config file $CONFIG_FILE not found, creating default configuration"
    cat > "$CONFIG_FILE" << EOF
app_name: chatbot
environment: $ENVIRONMENT
feature_flags:
  use_langgraph: true
  use_mistral: true
  enable_monitoring: true
  enable_cpu_optimizations: true
EOF
    log_info "Created default config file at $CONFIG_FILE"
fi

# Setup Python virtual environment
log_info "Setting up Python virtual environment in $VENV_DIR"
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip and install dependencies
log_info "Installing dependencies"
pip install --upgrade pip
pip install -e .

# Check if requirements.txt exists and install if it does
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Run tests based on environment
if [ "$ENVIRONMENT" == "production" ]; then
    log_info "Running critical tests for production deployment"
    python -m pytest tests/test_integration.py -v
elif [ "$ENVIRONMENT" != "development" ]; then
    log_info "Running all tests"
    python -m pytest
fi

# Check for MISTRAL_API_KEY
if [ -z "$MISTRAL_API_KEY" ]; then
    log_warn "MISTRAL_API_KEY environment variable not set"
    if [ "$ENVIRONMENT" == "production" ]; then
        log_error "MISTRAL_API_KEY is required for production environment"
        exit 1
    fi
fi

# Generate environment-specific settings
log_info "Generating environment settings for $ENVIRONMENT"
ENV_SETTINGS_DIR="./langgraph_integration/config"
mkdir -p $ENV_SETTINGS_DIR

cat > "$ENV_SETTINGS_DIR/env_settings.py" << EOF
"""
Environment-specific settings generated by deployment script.
This file is auto-generated - do not edit directly.
"""

ENVIRONMENT = "$ENVIRONMENT"
FEATURE_FLAGS = {
    "use_langgraph": True,
    "use_mistral": True if "$ENVIRONMENT" != "test" else False,
    "enable_monitoring": True,
    "enable_cpu_optimizations": True,
}

# Load custom settings from config file if needed
import os
import yaml

try:
    with open("$CONFIG_FILE", "r") as f:
        config = yaml.safe_load(f)
        if "feature_flags" in config:
            FEATURE_FLAGS.update(config["feature_flags"])
except:
    # Use default settings if config file can't be loaded
    pass

# Override with environment variables if present
for flag in FEATURE_FLAGS:
    env_var = f"ENABLE_{flag.upper()}"
    if env_var in os.environ:
        FEATURE_FLAGS[flag] = os.environ[env_var].lower() == "true"

# API keys
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
EOF

log_info "Environment settings generated at $ENV_SETTINGS_DIR/env_settings.py"

# Deploy based on environment
if [ "$ENVIRONMENT" == "development" ]; then
    log_info "Starting development server"
    python -m streamlit run langgraph_integration/streamlit_integration.py --server.port=$STREAMLIT_PORT
    
elif [ "$ENVIRONMENT" == "test" ]; then
    log_info "Test environment setup complete"
    
elif [ "$ENVIRONMENT" == "staging" ]; then
    log_info "Starting staging server"
    nohup python -m streamlit run langgraph_integration/streamlit_integration.py --server.port=$STREAMLIT_PORT > $LOG_DIR/streamlit.log 2>&1 &
    log_info "Streamlit server started on port $STREAMLIT_PORT (PID: $!)"
    
elif [ "$ENVIRONMENT" == "production" ]; then
    log_info "Deploying to production"
    # In a real production environment, you might use a process manager like supervisord or systemd
    # This is a simple example using nohup
    nohup python -m streamlit run langgraph_integration/streamlit_integration.py --server.port=$STREAMLIT_PORT > $LOG_DIR/streamlit.log 2>&1 &
    echo $! > $LOG_DIR/streamlit.pid
    log_info "Streamlit server started on port $STREAMLIT_PORT (PID: $!)"
    log_info "PID saved to $LOG_DIR/streamlit.pid"
fi

log_info "Deployment completed successfully for $ENVIRONMENT environment"

# Make script executable when created
chmod +x $0 