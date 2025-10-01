# Repository Structure

## `dataset_generation/`
Scripts for creating the synthetic dataset in JSON format. Depending on when you where in the process you want begin. You can use a small set of prompts to create more prompts. If you already have the prompts you want, you can generate the texts with the scripts. If you already have your texts, you can section them with the scripts. If you already have texts sectionned, you can label them with the scripts.

## `dataset_json_to_pytorch/`
Scripts for converting a JSON dataset into the Mistral-7B activations or residual stream and get them as PyTorch files. You need it to probe with `predictor_lazy.py`.

## `results/`
Contains our probing results of with the four configurations:

- `core_activation`  
- `core_residual`  
- `synthetic_activation`  
- `synthetic_residual`  

Each folder contains three CSV files:

- **`mazy`** → results of the true model  
- **`razy`** → results of the random model  
- **`mazy-razy`** → difference between the true model and random model results  

## `PHATE_Dimensional_Reduction`
Script for plotting dimensionality reduction using PHATE. You need a JSON dataset.

## `model.py`
Replacement for the existing model file in [`https://github.com/nickypro/taker/tree/main/src/taker`](https://github.com/nickypro/taker/tree/main/src/taker). You need this file to use `predictor_lazy.py`.

## `predictor_lazy.py`
The probing file. It need PyTorch activations or residual stream dataset and the update of the `model.py` file. It generate CSV result files

## `synthetic_dataset.json`
Synthetic dataset file.  
(**Note**: The CORE dataset is too large to upload.)

## `old_version_feb-2024/`
Contains files from the first iteration of the paper (February 2024).
