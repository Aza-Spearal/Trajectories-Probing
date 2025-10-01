# Repository Structure

## `dataset_generation/`
Scripts for creating the synthetic dataset.

## `dataset_json_to_pytorch/`
Scripts for converting JSON datasets into PyTorch-compatible datasets.

## `results/`
Contains the results of four configurations:

- `core_activation`  
- `core_residual`  
- `synthetic_activation`  
- `synthetic_residual`  

Each folder contains three CSV files:

- **`mazy`** → results of the true model  
- **`razy`** → results of the random model  
- **`mazy-razy`** → difference between the true model and random model results  

## `PHATE_Dimensional_Reduction`
Script for plotting dimensionality reduction using PHATE.

## `model.py`
Replacement for the existing model file in  
[`taker/tree/main/src/taker`](taker/tree/main/src/taker).

## `predictor_lazy.py`
Probes the dataset by loading it as a PyTorch dataset and generating CSV result files.

## `synthetic_dataset.json`
Synthetic dataset file.  
(**Note**: The CORE dataset is too large to upload.)

## `old_version_feb-2024/`
Contains files from the first iteration of the paper (February 2024).
