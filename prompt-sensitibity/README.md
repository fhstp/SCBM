# Prompt Sensitivity Experiments

Assess how persona/prompt variations affect the SCBM adjective-probability representation and downstream performance.

## Contents
- `get_persona_representation.py`: Compute adjective probabilities across multiple personas. 
- `inference_L31.py`: Lower-level LLaMA inference utilities.
- `persona-sensitivity.ipynb`: Notebook to analyze correlations and F1 changes across personas.

Current files in this folder:
- `README.md`
- `get_persona_representation.py`
- `inference_L31.py`
- `persona-sensitivity.ipynb`

## Data
Uses the same CSV format as elsewhere: `id`, `text`, optional `context`, `Class`.

## Usage 
- Ensure `HF_USER` and `HF_TOKEN` are set for first-time model cloning.
- Recommended: use `get_persona_representation.py` to generate persona-indexed feature pickles for the datasets used in the notebook (GermEval and TSNH). The script detects whether a dataset has `context` and switches prompts accordingly.

Example (fish):
```fish
set -x HF_USER your-username
set -x HF_TOKEN your-token
# GermEval (no context)
python get_persona_representation.py \
	--adjectives_path ../AdjectiveSetGeneration/adjectives.csv \
	--data_path ../Tasks/germeval/train.csv \
	--output_path ../Tasks/germeval

# TSNH (no context, cross-validation dataset; run per fold or on TSNH_uniform)
python get_persona_representation.py \
	--adjectives_path ../AdjectiveSetGeneration/adjectives.csv \
	--data_path ../Tasks/tsnh/TSNH_uniform.csv \
	--output_path ../Tasks/tsnh
```

Notes:
- Outputs are written as `diferent_persona_vectors.pickle` in the `--output_path` directory, containing for each persona index: `{ 'id': [...], 'text': [...], 'values': [probability_vector_per_instance] }`.
- The notebook `persona-sensitivity.ipynb` expects these pickles for analysis.
