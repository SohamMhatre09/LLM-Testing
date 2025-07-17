# 🚀 Quick Start Guide

## 1. Setup (First Time Only)
```bash
# Run the setup script to install dependencies
./setup.sh

# Activate the virtual environment
source llm_testing_env/bin/activate
```

## 2. Test a Single Model
```bash
# Edit the MODEL_NAME in llm_robustness_experiment.py
# Then run:
python llm_robustness_experiment.py
```

## 3. Quick Testing Multiple Models
```bash
# Edit the models list in test_models.py
# Then run:
python test_models.py
```

## 4. Batch Testing
```bash
# For comprehensive testing of multiple models:
python batch_test.py
```

## 5. Analyze Results
```bash
# After running experiments:
python analyze_results.py
```

## 📊 What Gets Generated

### After Each Test:
- `results_<model>_<timestamp>.csv` - Detailed trial data
- `analysis_<model>_<timestamp>.json` - Statistical summary

### After Analysis:
- `model_comparison_<timestamp>.csv` - Model comparison table
- `analysis_visualization_<timestamp>.png` - Charts and graphs
- `batch_testing_summary_<timestamp>.txt` - Batch test summary

## 🔧 Customization

### Change Models to Test:
Edit the `MODEL_NAME` variable in any of the scripts:
```python
MODEL_NAME = "gpt2"                    # GPT-2 base
MODEL_NAME = "distilgpt2"              # Smaller, faster
MODEL_NAME = "microsoft/DialoGPT-medium"  # Conversational
```

### Adjust Test Intensity:
```python
# Quick test (5 trials per variation)
experiment.run_experiment(trials_per_variation=5)

# Full test (100 trials per variation)  
experiment.run_experiment(trials_per_variation=100)
```

### Add New Prompt Injection Techniques:
1. Create new files in `resumes/poisoned/`
2. Follow the existing format with malicious instructions embedded

## 🎯 Expected Results

**Robust Model**: Low injection success rate (0-5%)
**Vulnerable Model**: High injection success rate (>20%)

The experiment tests 5 different prompt variations and 5 different injection techniques across temperature settings to provide comprehensive robustness assessment.
