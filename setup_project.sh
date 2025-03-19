#!/bin/bash
# Set up the QA Vision project from scratch

# Create necessary directories
mkdir -p data/screenshots output

# Create conda environment
if command -v conda &> /dev/null
then
    echo "Creating conda environment..."
    conda create -n qa_vision python=3.10 -y
    conda activate qa_vision
    
    # Install PyTorch with CUDA support
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        pip install torch torchvision
    else
        # Linux/Windows
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
    
    # Install dependencies
    pip install -e .
    pip install -e ".[training]"
else
    echo "Conda not found. Installing with pip..."
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -e .
    pip install -e ".[training]"
fi

# Login to Hugging Face (optional)
read -p "Do you want to log in to Hugging Face? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    huggingface-cli login
fi

# Make scripts executable
chmod +x scripts/*.py

echo "Project setup complete!"