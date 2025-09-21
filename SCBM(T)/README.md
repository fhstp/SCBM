# SCBM(T)

Training scripts for SCBM variants that operate on adjective-probability features stored in pickles next to the dataset CSVs.

- `SCBM.py`: SCBM with a lightweight classifier over 244-dim adjective features.
- `SCBMT.py`: SCBM-T variant that fuses text embeddings (e.g., XLM-R) with the adjective features.

## Expected inputs
For a dataset CSV like `../Tasks/hs_cs/train.csv`, there should be a corresponding pickle `train.csv.pickle` containing:
- `id`: list of ids
- `values`: list of 244-d vectors (adjective probabilities)

These are produced by the `Llama` pipeline or equivalent.

## Usage (example)
```fish
# Plain SCBM over features only
python SCBM.py --train_file_name ../Tasks/hs_cs/train.csv \
               --test_file_name ../Tasks/hs_cs/test.csv \
               --use_regularization false \
               --output_dir .

# SCBM-T (text + features)
python SCBMT.py --train_file_name ../Tasks/hs_cs/train.csv \
                --test_file_name ../Tasks/hs_cs/test.csv \
                --use_regularization false \
                --output_dir .
```

The scripts run multiple times, save the best checkpoint `SCBM*.pt` to `--output_dir`, and write results to a pickle.
