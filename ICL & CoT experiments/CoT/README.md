# Unified Parallel Batch Experiments

Scripts for running parallel batch experiments on different datasets using OpenAI's batch API.

> Note: For each experiment 5 runs are made to account for variability in LLM outputs and do majority voting. These runs outputs are saved in separate files so that when all parallel calls are completed the predictions can be aggregated.

## Scripts

- `batch_parallel_unified.py` - For conan, elf22, germeval, hscs datasets (multiple iterations)
- `batch_tsnh_unified.py` - For TSNH dataset (cross-validation folds)

### Standard Datasets (conan, elf22, germeval, hscs)

```bash
python "ICL & CoT experiments/CoT-python/batch_parallel_unified.py" --dataset hscs --output_dir "/home/rlabadie/SCBM/ICL & CoT experiments"
```

### TSNH Dataset

```bash
python "ICL & CoT experiments/CoT-python/batch_tsnh_unified.py" --output_dir "/home/rlabadie/SCBM/ICL & CoT experiments"
```

## Requirements

- Environment variable `OPENAI_API_KEY` must be set.

