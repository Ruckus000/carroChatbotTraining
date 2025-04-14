#!/usr/bin/env python
"""
Launcher script for running the LangGraph-enabled chatbot
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    """Main entry point for the launcher"""
    try:
        # Import streamlit first to avoid conflicts
        import streamlit as st
        
        # Then import our app
        from langgraph_integration.streamlit_integration import StreamlitApp
        
        print("Starting LangGraph-enabled chatbot...")
        app = StreamlitApp()
        app.render_chat_ui()
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please make sure you have installed all required packages:")
        print("pip install -e .")
        return 1
    except Exception as e:
        import traceback
        print(f"Error launching application: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 