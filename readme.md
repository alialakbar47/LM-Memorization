# LLM Data Extraction and MIA Evaluation

A comprehensive framework for evaluating membership inference attacks (MIA) on language models through data extraction using multiple scoring methods.

## Overview

This repository provides tools for:

1. **Data Extraction**: Generate text continuations using various scoring methods
2. **MIA Evaluation**: Assess the effectiveness of different scoring methods for membership inference attacks

## Features

- **Multiple Scoring Methods**: Implements various scoring approaches including likelihood, zlib compression, min-k, recall-based methods, and more
- **Flexible Generation**: Customizable text generation parameters
- **Comprehensive Evaluation**: Complete MIA evaluation with precision-recall metrics
- **Pipeline Automation**: Single command to run the complete pipeline

## Installation

```bash
git clone <repository-url>
cd llm-mia-evaluation
pip install -r requirements.txt
```

## Quick Start

### Complete Pipeline

Run the entire pipeline with default settings:

```bash
python run_pipeline.py --dataset_dir /path/to/datasets
```

### Custom Configuration

```bash
python run_pipeline.py \
    --dataset_dir /path/to/datasets \
    --model EleutherAI/gpt-neo-2.7B \
    --num_trials 10 \
    --val_set_num 500 \
    --temperature 0.7 \
    --top_p 0.9
```

## Usage

### Data Extraction Only

```bash
python extract.py \
    --dataset_dir /path/to/datasets \
    --model EleutherAI/gpt-neo-1.3B \
    --num_trials 5 \
    --val_set_num 1000
```

### MIA Evaluation Only

```bash
python evaluate_mia.py \
    --model EleutherAI/gpt-neo-1.3B \
    --guess_dir /path/to/guess/files
```

## Dataset Requirements

Your dataset directory should contain:

- `train_prefix.npy`: Prefix tokens for generation
- `train_dataset.npy`: Complete training dataset for evaluation
- `non_member_prefix.npy` (optional): Non-member prefixes for original recall calculation

## Scoring Methods

The framework implements the following scoring methods:

### Base Methods
- **Likelihood**: Standard log-likelihood scoring
- **Zlib**: Compression-based scoring
- **Metric**: Likelihood with outlier removal
- **High Confidence**: Confidence-adjusted scoring

### Recall Methods
- **Recall**: Conditional vs unconditional likelihood
- **Recall2/3**: Variations of recall scoring
- **Original Recall**: Using non-member prefixes

### Min-k Methods
- **Min-k**: Bottom-k token probabilities
- **Min-k++**: Normalized min-k scoring

## Configuration Options

### Generation Parameters

- `--top_k`: Top-k sampling (default: 24)
- `--top_p`: Top-p sampling (default: 0.8)
- `--temperature`: Sampling temperature (default: 0.58)
- `--repetition_penalty`: Repetition penalty (default: 1.04)

### Experiment Settings

- `--num_trials`: Number of generation trials per prompt (default: 5)
- `--val_set_num`: Number of validation examples (default: 1000)
- `--batch_size`: Processing batch size (default: 64)
- `--seed`: Random seed for reproducibility (default: 2022)

## Output Files

### Extraction Output
- `tmp/experiment_name/generations/`: Generated sequences
- `tmp/experiment_name/losses/`: Scoring method results
- `guess_*.csv`: Guess files with ground truth labels

### MIA Evaluation Output
- `results/mia_evaluation/`: MIA evaluation results
- Metrics include AUROC, FPR95, TPR05, Average Precision, and more

## Pipeline Control

### Skip Steps

```bash
# Skip extraction, only run MIA evaluation
python run_pipeline.py --skip_extraction --guess_dir /path/to/existing/guesses

# Skip MIA evaluation, only run extraction
python run_pipeline.py --skip_mia
```

## File Structure

```
.
├── extract.py              # Main extraction script
├── evaluate_mia.py         # MIA evaluation script
├── utils.py                # Utility functions
├── run_pipeline.py         # Complete pipeline runner
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Advanced Usage

### Custom Model

```bash
python run_pipeline.py \
    --model /path/to/custom/model \
    --dataset_dir /path/to/datasets
```

### Large Scale Experiments

```bash
python run_pipeline.py \
    --num_trials 20 \
    --val_set_num 5000 \
    --batch_size 128 \
    --experiment_name large_scale_experiment
```

### Evaluation Only with Existing Data

```bash
python evaluate_mia.py \
    --model EleutherAI/gpt-neo-1.3B \
    --guess_dir guess_files/
```

## Performance Tips

- Use larger batch sizes for faster processing on GPUs with more memory
- Adjust `num_trials` based on your computational budget
- Use FP16 models for faster inference (automatically enabled)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm-mia-evaluation,
  title={LLM Data Extraction and MIA Evaluation Framework},
  year={2024},
  howpublished={\url{https://github.com/your-repo/llm-mia-evaluation}}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
