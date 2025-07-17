#!/bin/bash

# LLM Robustness Testing - Quick Run Script
echo "🧪 LLM Robustness Testing Setup & Run"
echo "====================================="

# Check if virtual environment exists
if [ ! -d "llm_testing_env" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv llm_testing_env
    
    # Activate virtual environment
    source llm_testing_env/bin/activate
    
    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    echo "📚 Installing dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers accelerate datasets numpy pandas scipy scikit-learn matplotlib seaborn
    
    echo "✅ Environment setup complete!"
else
    echo "✅ Virtual environment already exists"
    source llm_testing_env/bin/activate
fi

# Check if resume folders exist
if [ ! -d "resumes" ]; then
    echo "❌ Error: 'resumes' folder not found!"
    echo "Please make sure you have the resume folders with the test data."
    exit 1
fi

# Run the experiment
echo "🚀 Starting LLM robustness experiment..."
python main.py

echo ""
echo "🏁 Experiment complete! Check the generated CSV and JSON files for results."
