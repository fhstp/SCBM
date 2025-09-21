# Zero-shot Evaluation

Zero-shot classification with OpenAI GPT models and LLaMA-3.1 pipelines.

## Folders
- `openai-zero-shot.py`: OpenAI Chat Completions, 4 passes per item with majority-style analysis saved to pickle.
- `llama-zero-shot.py`: Local LLaMA-3.1 pipeline; runs 4 passes per item and pickles results.

## Inputs
CSV with:
- `id`, `text`, optional `context`, `Class` (used to derive the class set).

## Environment
- OpenAI: `OPENAI_API_KEY` must be set.
- LLaMA: set `HF_USER` and `HF_TOKEN` to clone on first run.

## Usage
OpenAI example:
```fish
set -x OPENAI_API_KEY "sk-..."
python openai-zero-shot.py --file_path ../Tasks/conan/test.csv --model chatgpt-4o-latest
```

LLaMA example:
```fish
set -x HF_USER your-username
set -x HF_TOKEN your-token
python llama-zero-shot.py --file_path ../Tasks/germeval/test.csv
```

Outputs: `file.csv.gpt35.pickle` or `file.csv.llama31.pickle` with multiple runs.
