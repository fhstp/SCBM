# AdjectiveDefinition

Generate short definitions for a curated list of adjectives using OpenAI Chat Completions.

## What this does
- Reads `adjectives_sortd_en.csv` with a column `adjective`.
- Prompts an LLM to produce a one-line definition per adjective.
- Writes `adjectives_with_definitions.csv` alongside the input file.

## Requirements
- Environment variable `OPENAI_API_KEY` must be set.

## Usage
From this folder:

```fish
# export your OpenAI key (fish shell)
set -x OPENAI_API_KEY "sk-..."

# run (default uses adjectives_sortd_en.csv)
python get_definitions.py

# or specify a custom path
python get_definitions.py --adjectives_path ./adjectives_sortd_en.csv
```

## Input/Output
- Input: `adjectives_sortd_en.csv` with column `adjective`.
- Output: `adjectives_with_definitions.csv` in the same folder.
