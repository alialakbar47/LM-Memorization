# Quick Start Guide for Refactored Repository

## What's New?

The repository has been refactored to use a modular, configuration-driven architecture. **All existing functionality remains the same**, but the code is now more organized and easier to customize.

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install pyyaml  # New dependency for config files
```

### 2. Try the Example

```bash
python example_usage.py
```

This will:

- Load the configuration from `config.yaml`
- Load all enabled metrics
- Run them on sample data
- Show you the results

### 3. Customize Your Experiment

Edit `config.yaml` to:

- Enable/disable metrics
- Change hyperparameters
- Adjust model and dataset settings

```yaml
metrics:
  likelihood:
    enabled: true # ← Enable/disable here

  min_k_0.2:
    enabled: true
    config:
      k_ratio: 0.2 # ← Adjust parameters here
```

## Usage Patterns

### Pattern 1: Use Existing Scripts (No Changes Needed)

Your existing commands still work:

```bash
python run_pipeline.py --dataset_dir datasets/ --model EleutherAI/gpt-neo-1.3B
```

### Pattern 2: Use Configuration File (Recommended)

```bash
# Edit config.yaml, then:
python run_pipeline.py --config config.yaml
```

### Pattern 3: Python API

```python
from metric_loader import load_config, load_all_enabled_metrics
from utils import load_model_and_tokenizer

config = load_config('config.yaml')
model, tokenizer = load_model_and_tokenizer(config['global']['model'])
metrics = load_all_enabled_metrics(model, tokenizer, config)

# Use metrics
for name, metric in metrics.items():
    scores = metric.compute_score(tokens)
```

## What Files Were Added?

### New Files

- `metrics/` - Directory with individual metric implementations
  - `__init__.py` - Base class for all metrics
  - `likelihood.py`, `zlib.py`, etc. - Individual metrics
- `config.yaml` - Configuration file for experiments
- `metric_loader.py` - Utility to load metrics from config
- `example_usage.py` - Example showing how to use new system
- `REFACTORING_GUIDE.md` - Detailed documentation
- `QUICKSTART.md` - This file

### Unchanged Files

- `utils.py` - Original utility functions (still works)
- `extract.py` - Original extraction script (still works)
- `evaluate_mia.py` - Original MIA evaluation (still works)
- `run_pipeline.py` - Original pipeline (still works)

## Advantages of New Structure

### ✅ Easy Experimentation

Change metrics and hyperparameters by editing YAML, not code.

### ✅ Modular Design

Each metric is self-contained in its own file.

### ✅ Clear Configuration

All settings in one place, easy to share and version control.

### ✅ Backward Compatible

Existing scripts continue to work without modifications.

### ✅ Extensible

Add new metrics by creating a new file and updating config.

## Common Tasks

### Enable/Disable Metrics

Edit `config.yaml`:

```yaml
metrics:
  likelihood:
    enabled: true # Enable

  zlib:
    enabled: false # Disable
```

### Change Hyperparameters

Edit `config.yaml`:

```yaml
metrics:
  metric:
    config:
      num_std: 3 # Change from 3 to 2
```

### Add a New Metric

1. Create `metrics/my_metric.py`:

```python
from metrics import AbstractMetric

class MyMetric(AbstractMetric):
    def compute_score(self, generated_tokens, **kwargs):
        # Your logic here
        return scores
```

2. Add to `config.yaml`:

```yaml
metrics:
  my_metric:
    module: "my_metric"
    enabled: true
    config: {}
```

### Run Different Experiments

```bash
# Create experiment-specific configs
cp config.yaml configs/experiment1.yaml
cp config.yaml configs/experiment2.yaml

# Edit each config file
# Run experiments
python run_pipeline.py --config configs/experiment1.yaml
python run_pipeline.py --config configs/experiment2.yaml
```

## Troubleshooting

### Missing YAML module

```bash
pip install pyyaml
```

### Metric not loading

Check that:

1. Module name in config matches filename in `metrics/`
2. Class name follows convention: `<ModuleName>Metric`
3. Metric is marked as `enabled: true`

### Want to use old approach?

Just use the original commands without `--config`:

```bash
python run_pipeline.py --dataset_dir datasets/
```

## Learning More

- **Quick overview**: This file
- **Detailed guide**: [`REFACTORING_GUIDE.md`](REFACTORING_GUIDE.md)
- **Example code**: [`example_usage.py`](example_usage.py)
- **Configuration**: [`config.yaml`](config.yaml)

## Need Help?

1. Check [`REFACTORING_GUIDE.md`](REFACTORING_GUIDE.md) for detailed documentation
2. Run `example_usage.py` to see metrics in action
3. Look at existing metrics in `metrics/` directory for examples
4. Your existing scripts still work - no pressure to migrate immediately!

## Summary

**The refactoring adds powerful new features while keeping everything backward compatible.**

- ✨ New: Configuration-driven, modular metric system
- ✅ Old: All existing scripts and workflows still work
- 🎯 You choose: Use new system when ready, keep using old approach if preferred

Happy experimenting! 🚀
