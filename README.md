# Project Structure

`dataset_generation/`
Scripts for creating the synthetic dataset.

`dataset_json_to_pytorch/`
Scripts for converting JSON datasets into PyTorch-compatible datasets.

### results
Contains the result of the 4 possibilities:
- **core_activation**
- **core_residual**
- **synthetic_activation**
- **synthetic_residual**

Each of these contains 3 CSV files:
- **mazy** → shows the results of the true model.  
- **razy** → shows the results of the random model.  
- **mazy-razy** → shows the difference between the true model results and the random model results.  

### PHATE_Dimensional_Reduction
This file plots the dimensional reduction with PHATE.

### model.py
Should replace the one in  
[`taker/tree/main/src/taker`](https://github.com/nickypro/taker).  

### predictor_lazy.py
This file probes the dataset.  
It uses the PyTorch dataset and creates the CSV files as results.

### synthetic_dataset.json
This is the synthetic dataset file.  
(The CORE dataset is too big to be uploaded.)

### old_version_feb-2024
Contains files used for the first iteration of the paper (February 2024).
