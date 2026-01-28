#!/bin/bash
# Setup script for LLM CV Analysis Pipeline

echo "Setting up LLM CV Analysis Pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To run the analysis:"
echo "  source venv/bin/activate"
echo "  python run_analysis.py --quick-test"

