# Transformers Baseline

Train and evaluate transformer baselines (BERT/XLM-R) on the tasks, with single split or 5-fold cross-validation.

## Scripts
- `run_transformers.py`: Train on train.csv and evaluate on dev.csv; repeat 5 runs and save F1s.
- `run_transformers-crossval.py`: 5-fold CV on `train.csv`, repeated per model; save F1s.
- `models.py`: Model definitions and helper routines.

## Data format
CSV files with columns:
- `id`
- `text`
- `Class` (label)
- Optional `context` (some datasets)

## Usage
Single train/dev split:

```fish
python run_transformers.py --train_file ../Tasks/elf22/train.csv \
                           --dev_file ../Tasks/elf22/test.csv \
                           --output_file ./elf22_baselines.pickle
```

Cross-validation (e.g., TSNH):

```fish
python run_transformers-crossval.py --train_file ../Tasks/tsnh/TSNH_uniform.csv \
                                    --output_file ./tsnh_cv_baselines.pickle
```

Outputs: a pickle mapping model name -> list of F1 scores across runs.
