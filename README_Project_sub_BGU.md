# test_downprompt_simple.py

## Overview

This script provides a **clean, reproducible evaluation pipeline** for node classification on Cora and Citeseer using the MultiGPrompt framework. It is designed to **avoid data leakage** and supports both prototype-based and simple MLP classifier evaluation modes. All configuration, data loading, model loading, and evaluation are handled in a single script.

## Features

- **No data leakage**: Test nodes are isolated from training/validation during evaluation.
- **Flexible evaluation**: Choose between prototype-based (downprompt) and simple MLP classifier modes.
- **Deterministic sampling**: Supports balanced and reproducible test set sampling.
- **Single or multi-run**: Run a single evaluation or multiple runs with different seeds.
- **Minimal output**: Only essential results and errors are printed.

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data and models

- Place the required `.pkl` dataset/model files in the `../modelset/` directory.
- Supported datasets: `cora`, `citeseer`.

### 3. Run the script

```bash
python test_downprompt_simple.py
```

By default, the script uses the configuration in `config.py` (e.g., dataset, evaluation mode, sample size, etc.).

#### To run a custom evaluation:

You can call the `run_custom_evaluation` function from within the script or an interactive session:

```python
from test_downprompt_simple import run_custom_evaluation
accuracy, preds, trues, n_classes, model_loaded = run_custom_evaluation(
    dataset='cora', sample_size=50, seed=42, use_all_test_data=False, single_run_mode=True
)
```

### 4. Configuration

Edit `config.py` to change:
- Dataset (`'cora'` or `'citeseer'`)
- Evaluation mode (`use_simple_classifier`)
- Sample size, random seed, and other hyperparameters

### 5. Output

- Prints final accuracy, per-class statistics, and summary.
- Optionally saves confusion matrix plots if enabled in the code.

## Requirements

- torch==1.10.1
- scikit-learn
- numpy
- scipy
- tqdm

(See `requirements.txt` for details.)

## Notes

- The script is self-contained and does not require training from scratch if pre-trained models are available.
- All debug and verbose prints have been removed for clarity.
- For best reproducibility, set the random seed in `config.py`. 