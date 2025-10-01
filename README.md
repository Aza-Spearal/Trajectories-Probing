# Repository Structure

## `dataset_generation/`
Scripts to create the synthetic dataset in JSON format. Depending on what stage you are at, you can:
- Use a small set of prompts to expand and generate more prompts with the script `generate_dataset.ipynb`.
- If you already have prompts, generate texts from them with the script `generate_dataset.ipynb`.
- If you already have texts, section them with the script `chunk_label_GPT-4.ipynb`.
- If you already have sectioned texts, label them with the script `chunk_label_GPT-4.ipynb`.

## `dataset_json_to_pytorch/`
Scripts to convert a JSON dataset into Mistral-7B activations or the residual stream, and save them as PyTorch files. These files are required by `predictor_lazy.py` for probing.

## `results/`
Contains probing results for the four configurations:

- `core_activation`  
- `core_residual`  
- `synthetic_activation`  
- `synthetic_residual`  

Each folder contains three CSV files:

- **`mazy`** → results of the true model  
- **`razy`** → results of the random model  
- **`mazy-razy`** → difference between the true model and random model results  

## `PHATE_Dimensional_Reduction`
Script for plotting dimensionality reduction using PHATE. Requires a JSON dataset.

## `model.py`
Replacement for the existing model file in the `taker` repo ([`https://github.com/nickypro/taker/tree/main/src/taker`](https://github.com/nickypro/taker/tree/main/src/taker)). This update is required to run `predictor_lazy.py`.

## `predictor_lazy.py`
The probing file. It need PyTorch activations or residual stream dataset and the update of the `model.py` file. It generate CSV result files

Probing script. It requires PyTorch activations or residual stream files (from `dataset_json_to_pytorch/`) and the updated `model.py`. The script generates CSV result files that you can see in the `results/` folders.

## `synthetic_dataset.json`
Synthetic dataset file.  
(**Note**: The CORE dataset is too large to upload.)

## `old_version_feb-2024/`
Contains files from the first iteration of the paper (February 2024).
