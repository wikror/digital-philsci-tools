#!/bin/bash
# Setup script for BERTopic analysis environment on Apple Silicon (M4 Pro)
# This script creates and configures the conda environment with GPU support

set -e  # Exit on error

echo "=================================="
echo "BERTopic Environment Setup"
echo "=================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Environment name
ENV_NAME="bertopic"
ENV_FILE="environment_bertopic.yml"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found in current directory"
    exit 1
fi

echo "📋 Environment file: $ENV_FILE"
echo ""

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "   Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "ℹ️  Updating existing environment..."
        conda env update -n $ENV_NAME -f $ENV_FILE --prune
        ENV_CREATED=false
    fi
fi

# Create environment if needed
if [ "${ENV_CREATED:-true}" = true ]; then
    echo "🔨 Creating conda environment from $ENV_FILE..."
    conda env create -f $ENV_FILE
fi

echo ""
echo "✅ Conda environment created/updated successfully"
echo ""

# Activate environment and install additional resources
echo "📦 Installing additional NLP resources..."
echo ""

# Use conda run to execute in the environment
conda run -n $ENV_NAME python -m spacy download en_core_web_sm --quiet

echo "✅ Spacy model downloaded"

conda run -n $ENV_NAME python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo "✅ NLTK data downloaded"
echo ""

# Verify GPU availability
echo "🔍 Checking GPU availability..."
echo ""

conda run -n $ENV_NAME python << EOF
import torch
import platform

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"")

if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) is available!")
    print("   Your M4 Pro GPU can be used for acceleration.")
    print(f"   MPS built: {torch.backends.mps.is_built()}")
else:
    print("⚠️  MPS is not available.")
    print("   Models will run on CPU only.")

print(f"")
print(f"CPU cores: {torch.get_num_threads()}")
EOF

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify the installation, run:"
echo "  python -c \"from topic_modeling_analysis import *; print('✅ Module loaded successfully')\""
echo ""
echo "To run the Jupyter notebook:"
echo "  jupyter lab topic_modeling_notebook.ipynb"
echo ""
echo "For GPU usage tips, see: README_ENVIRONMENT.md"
echo ""
