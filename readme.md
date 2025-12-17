# LLM Data Extraction and MIA Evaluation - Refactored

This is a refactored version of the LLM data extraction and membership inference attack (MIA) evaluation framework. The code has been restructured for better modularity and ease of extension.

## Key Improvements

### 1. Modular Metrics System
Metrics are now organized in a dedicated `metrics/` folder with each metric in its own file:

```
metrics/
├── __init__.py          # Metric registry
├── base.py              # Base metric class
├── likelihood.py        # Likelihood metric
├── zlib.py             # Zlib metric
├── metric.py           # Metric score
├── high_confidence.py  # High confidence metric
├── recall.py           # Recall-based metrics
├── lowercase.py        # Lowercase metric
└── min_k.py            # Min-k based metrics
```

### 2. YAML Configuration
All hyperparameters are now managed through YAML config files instead of command-line arguments:

```yaml
experiment:
  name: my_experiment
  seed: 2022
  dataset_dir: ../datasets

model:
  name: EleutherAI/gpt-neo-1.3B

generation:
  num_trials: 5
  val_set_num: 1000
  batch_size: 64
  ...
```

### 3. Organized Utilities
Utilities are split into logical modules:

```
utils/
├── core.py          # Core utilities (seeds, model loading)
├── data.py          # Data loading/saving
├── evaluation.py    # Evaluation metrics
└── checkpoint.py    # Checkpoint management
```

## Project Structure

```
.
├── configs/
│   └── extraction_default.yaml   # Example config file
├── metrics/
│   ├── __init__.py
│   ├── base.py
│   ├── likelihood.py
│   ├── zlib.py
│   ├── metric.py
│   ├── high_confidence.py
│   ├── recall.py
│   ├── lowercase.py
│   └── min_k.py
├── utils/
│   ├── core.py
│   ├── data.py
│   ├── evaluation.py
│   └── checkpoint.py
├── config.py                      # Configuration management
├── extract.py                     # Main extraction script
├── evaluate_mia.py               # MIA evaluation script
├── run_pipeline.py               # Full pipeline runner
└── README.md
```

## Usage

### Using Config Files (Recommended)

1. **Create a config file** (see `configs/extraction_default.yaml` for example)

2. **Run extraction:**
```bash
python extract.py --config configs/my_config.yaml
```

3. **Run MIA evaluation:**
```bash
python evaluate_mia.py --config configs/mia_config.yaml
```

4. **Run full pipeline:**
```bash
python run_pipeline.py --config configs/my_config.yaml
```

### Using Command-Line Arguments

You can still override config values with command-line arguments:

```bash
python extract.py --config configs/default.yaml --num_trials 10 --batch_size 32
```

Or run without a config file (uses defaults):

```bash
python extract.py --experiment_name my_exp --model EleutherAI/gpt-neo-1.3B --num_trials 5
```

### Resume from Checkpoint

```bash
python extract.py --config configs/my_config.yaml --resume
```

### Force Restart

```bash
python run_pipeline.py --config configs/my_config.yaml --force_restart
```

## Adding New Metrics

To add a new metric, follow these steps:

1. **Create a new file** in `metrics/` (e.g., `my_metric.py`)

2. **Implement the metric class:**

```python
from .base import BaseMetric
import numpy as np

class MyMetric(BaseMetric):
    """My custom metric."""
    
    def __init__(self, **kwargs):
        super().__init__(name="my_metric", **kwargs)
    
    def compute(self, model, tokenizer, generated_tokens, 
                outputs, device, **kwargs) -> np.ndarray:
        """Compute your metric scores."""
        # Your implementation here
        scores = []
        # ... compute scores ...
        return np.array(scores)
    
    def direction(self) -> str:
        """Return 'min' or 'max' for optimization direction."""
        return "max"  # or "min"
```

3. **Register the metric** in `metrics/__init__.py`:

```python
from .my_metric import MyMetric

METRIC_REGISTRY = {
    # ... existing metrics ...
    'my_metric': MyMetric,
}

def get_all_metrics(...):
    metrics = [
        # ... existing metrics ...
        MyMetric(),
    ]
    return metrics
```

4. **Use the metric:**

```bash
python extract.py --config configs/my_config.yaml --save_all_methods
```

Your new metric will automatically be:
- Computed during extraction
- Saved to CSV files
- Evaluated in MIA evaluation

## Configuration Options

### Experiment Settings
- `name`: Experiment name (creates `results/{name}/`)
- `seed`: Random seed for reproducibility
- `dataset_dir`: Path to dataset directory

### Model Settings
- `name`: HuggingFace model name or path

### Generation Settings
- `num_trials`: Number of generation trials per prompt
- `val_set_num`: Number of validation examples to use
- `batch_size`: Batch size for processing
- `top_k`, `top_p`, `temperature`: Generation parameters
- `typical_p`: Typical sampling parameter
- `repetition_penalty`: Repetition penalty

### Saving Settings
- `save_all_generations_per_prompt`: Save CSVs for all generation tiers
- `save_all_methods`: Save CSVs for all metrics (not just likelihood)
- `save_npy_files`: Save intermediate .npy files

### Checkpoint Settings
- `save_checkpoints`: Enable checkpointing
- `resume`: Resume from checkpoint
- `force_restart`: Remove existing checkpoint and restart

### Metric Settings
- `suffix_len`: Length of suffix for metrics (default: 50)
- `prefix_len`: Length of prefix (default: 50)
- `k_ratios`: Ratios for min-k metrics
- `max_entropy`: Maximum entropy threshold for surprise metric

## Output Structure

```
results/
└── {experiment_name}/
    ├── config.yaml                      # Saved configuration
    ├── checkpoint.pkl                   # Checkpoint (if enabled)
    ├── extraction_metrics_summary.csv   # Extraction results
    ├── generations/                     # Generated sequences (optional)
    ├── losses/                          # Loss values (optional)
    ├── guess_files/                     # CSV files for MIA
    │   ├── guess_likelihood_5.csv
    │   └── ...
    └── mia_evaluation/                  # MIA results
        └── {model_name}_mia_results.csv
```

## Differences from Original

### What Changed
1. **File Organization**: Metrics moved to `metrics/`, utilities to `utils/`
2. **Configuration**: YAML configs instead of command-line arguments
3. **Modularity**: Each metric is a separate class with clear interface
4. **Registry Pattern**: Metrics registered in `__init__.py` for easy discovery

### What Stayed the Same
1. **All computation logic**: Identical metric implementations
2. **Results**: Produces exactly the same results as original code
3. **Checkpoint system**: Same checkpointing behavior
4. **Output format**: Same CSV and file structure

## Requirements

```
torch
transformers
numpy
pandas
scikit-learn
pyyaml
tqdm
```

## License

Same as original repository.
