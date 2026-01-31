#!/bin/bash

# Setup script for AI Prompt Security Detection System
# This script sets up the Python environment and downloads datasets

echo "=========================================="
echo "AI PROMPT SECURITY - SETUP SCRIPT"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✅ Python detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
echo "⚠️  This may take several minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ All dependencies installed"
echo ""

# Create __init__.py files
touch src/__init__.py

echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download datasets:"
echo "   python src/download_datasets.py"
echo ""
echo "3. Clean and process data:"
echo "   python src/data_cleaning.py"
echo ""
echo "=========================================="
