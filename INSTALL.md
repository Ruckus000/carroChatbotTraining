# Installation Guide

This guide provides instructions for setting up the Chatbot Training Framework.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd carroChatbotTraining
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r chatbot/requirements.txt
   ```

### Apple Silicon (M1/M2) Specific Instructions

If you're using Apple Silicon (M1/M2 chip), you might need to install PyTorch using a specific command to get GPU acceleration with Metal:

```bash
pip install torch torchvision torchaudio
```

This will install the version of PyTorch that's compatible with Apple's Metal Performance Shaders (MPS).

## Verify Installation

To verify your installation, run:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('MPS available:', getattr(torch, 'has_mps', lambda: False)())"
```

This should display the PyTorch version and whether CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon) is available.

## Data Preparation

Before training, you need to prepare a JSON file with your conversation data. See the README.md file for the required format.

## Running the Framework

Once installed, you can run the framework with:

```bash
python chatbot/chatbot_training.py --input_data path/to/conversations.json --output_dir ./output --augment_data --train_models
```

For more information on usage and options, see the README.md file.
