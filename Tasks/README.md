# Tasks

Datasets used across the repository with a common CSV schema and folder-specific notes.

## Common columns
- `id`: unique identifier
- `text`: the comment/text
- `Class`: label name
- Optional `context`: extra text context for some datasets

## Subfolders
- `conan/`, `elf22/`, `germeval/`, `hs_cs/`: have `train.csv` and `test.csv`.
- `tsnh/`: provides `TSNH_uniform.csv` and a 5-fold split under `dataset_5folding/`.

