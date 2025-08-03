# In-Context Learning (ICL) Experiments

This directory contains Jupyter notebooks for conducting In-Context Learning experiments across multiple hate speech and counter speech detection datasets.

## Overview

These notebooks implement ICL experiments where examples from training data are used to prompt large language models (LLMs) for few-shot classification. Each notebook follows a two-step process:

1. **Example Selection**: Automatically selects representative examples from training data for each class
2. **ICL Inference**: Uses selected examples to prompt GPT-4o for classification on test data

## Notebooks

- `ICL-CONAN.ipynb` - Counter-narrative detection (facts, support, denouncing, hypocrisy, unrelated, humor, question)
- `ICL-ELF22.ipynb` - Troll vs counterspeech classification
- `ICL-GERMEVAL.ipynb` - Offensive language detection (German text)
- `ICL-HS_CS.ipynb` - Hate speech vs counterspeech vs neither classification
- `ICL-TSNH.ipynb` - Counter speech detection (binary classification)

## Methodology

### Example Selection Process
- Filters training examples by text length (< 180 characters for readability)
- Randomly samples 3-5 examples per class
- Formats examples with special markers (`<text_icl_begin>` ... `<text_icl_end>`)
- Saves formatted prompts to `../icl_promtps/{dataset}.json`

### ICL Inference Process
- Loads pre-generated ICL prompts
- For each test instance, creates a prompt with examples + new instance
- Queries GPT-4o model (`chatgpt-4o-latest`) with max 10 tokens
- Runs 4 iterations per dataset for statistical significance
- Saves results to `outputs/{dataset}/test_{i}.csv`

### Evaluation
- Calculates F1-macro scores across all iterations
- Reports mean and standard deviation of performance

- OpenAI API key in environment (`.env` file)
- Python packages: `pandas`, `openai`, `scikit-learn`, `tqdm`
- Access to training/test datasets in `../Tasks/` directory
