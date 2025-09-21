# AdjectiveSetGeneration

Iteratively generate a broad set of adjectives relevant to counterspeech recognition using OpenAI, then compare coverage vs expert annotated gold adjective set.

## What this does
- Seeds an adjective list via an LLM, then extends it for N iterations.
- Optionally evaluates overlap against a gold standard adjective list.

## Requirements
- `OPENAI_API_KEY` in the environment.
- For WordNet lemmatization and synonyms install NLTK data once:
  ```python
  import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')
  ```

## Usage
From this folder:

```fish
set -x OPENAI_API_KEY "sk-..."
# Generate adjectives (4 iterations by default) and write adjectives.csv
python adjectives_generation.py --output_file adjectives.csv --generation_iterations 4


Notes:
- The script already persists all per-iteration lists to `adjectives-iterations.pkl` and the final merged list to `adjectives.csv`.
