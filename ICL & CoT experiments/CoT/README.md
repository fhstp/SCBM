# Unified Parallel Batch Experiments

Unified scripts for running parallel batch experiments on different datasets using OpenAI's batch API.

## Scripts

- `batch_parallel_unified.py` - For conan, elf22, germeval, hscs datasets (multiple iterations)
- `batch_tsnh_unified.py` - For TSNH dataset (cross-validation folds)

## Usage

### Standard Datasets (conan, elf22, germeval, hscs)

```bash
python "ICL & CoT experiments/CoT-python/batch_parallel_unified.py" --dataset hscs --output_dir "/home/rlabadie/SCBM/ICL & CoT experiments"
```

### TSNH Dataset

```bash
python "ICL & CoT experiments/CoT-python/batch_tsnh_unified.py" --output_dir "/home/rlabadie/SCBM/ICL & CoT experiments"
```

## Requirements

- OpenAI API key in environment or `.env` file
- Python packages: `pandas`, `scikit-learn`, `openai`, `fire`
