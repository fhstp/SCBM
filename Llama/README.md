# Llama

Compute SCBM adjective-probability representations with LLaMA-3.1 and run inference with and without context.

## Scripts
- `main.py`: Unified CLI to run inference over CSVs using `inference.py` utilities.
- `inference.py`: Loads LLaMA-3.1, builds prompts, and returns probabilities for each adjective.

## Data expectations
Input CSV(s) must contain at least:
- `id`: unique identifier
- `text`: the comment text
- Optional `context`: context string (only needed when using `--use_context true`)

The adjectives file is a CSV with a column `adjective`, e.g. `../AdjectiveSetGeneration/adjectives.csv`.

## Hugging Face auth
Scripts clone the model from Hugging Face using `HF_USER` and `HF_TOKEN` from the environment. Set them for first run:

```fish
set -x HF_USER your-username
set -x HF_TOKEN your-token
```

## Examples
Run no-context inference over one file:

```fish
python main.py --input_files ../Tasks/germeval/test.csv \
               --use_context false \
               --adjectives_file ../AdjectiveSetGeneration/adjectives.csv \
               --repository meta-llama/Llama-3.1-8B-Instruct \
               --batch_size 244
```

Run context inference over multiple files:

```fish
python main.py --input_files "[\"../Tasks/hs_cs/test.csv\", \"../Tasks/hs_cs/train.csv\"]" \
               --use_context true \
               --adjectives_file ../AdjectiveSetGeneration/adjectives.csv
```

Outputs are pickles saved next to the input file, e.g. `test.csv.pickle` with fields:
- `id`: list of ids
- `values`: list of probability vectors (len = num adjectives)
- `text` (for no-context runs)
