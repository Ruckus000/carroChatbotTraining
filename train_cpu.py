#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPU-only wrapper for training the chatbot models.
This script forces CPU usage before importing any torch modules.
"""

# Force CPU usage before importing any other modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Explicitly disable MPS
import sys
sys.argv.insert(1, "--no_mps")

# Run the main training script
from chatbot_training import main

if __name__ == "__main__":
    main() 