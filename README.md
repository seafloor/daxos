# DAXOS: Dask and XGBoost on SNPs

DAXOS is a Python library designed to efficiently run Dask and XGBoost on SNP data, particularly for use on high-performance computing (HPC) systems that use a SLURM management system. The code is optimized to reduce memory usage and is tailored for handling genetic data from PLINK files.

## Features

- **Memory-efficient processing:** Utilizes Dask to handle large datasets with reduced memory usage.
- **XGBoost integration:** Leverages the power of XGBoost for fast and scalable machine learning.
- **HPC optimization:** Designed specifically for HPC environments.
- **PLINK to HDF5 conversion:** Includes convenience functions to convert PLINK files to HDF5 format.
- **Covariate adjustment:** Functions for adjusting covariates in genetic data before and after analysis
- **Cross-validation and refitting:** Support for model cross-validation, refitting, and prediction, and explanation through SHAP values

## Installation

To install the required dependencies, use the provided `requirements.yaml` file with conda:

```sh
conda env create -f requirements.yaml
conda activate daxos
```

## Usage

Example SLURM submission scripts are included - please edit for your own file paths, module names etc. All submission scripts start with "submit", and often call python scipts which pull together all the functions in the main package.

### PLINK to HDF5 Conversion

Convert PLINK files to HDF5 format for more efficient processing using the `submit_plink_to_hdf5.sh` file. Note that you need plink installed, as well as the `shuf` unix tool (usually already available) on linux systems.

### Running Cross-Validation and Refitting

Perform cross-validation and refit the model on the genetic data:

```python
from daxos.crossvalidate import run_cv_and_refit

# Example usage
run_cv_and_refit(client, X, y, params)
```

### Covariate Adjustment

Adjust for covariates in your genetic data before analysis using the script `submit_adjust_hdf5_for_covars.sh`. After analysis, covariate adjusted predictions in the test set can be generated using daxos/scoring.py, which runs OLS or RF (preferred) adjustment for covariates. The latter is important to adjust for the non-linear effects of covariates when we're running non-linear prediction models.

### Training and predicting from models
The script `submit_daxg.sh` is supposed to be a helper script, but it's long and complicated for a slurm submission script. You can also just called the cv.py, refit.py or predict.py scripts with your own submission script.

The good thing it does have is something that prints the node and tunneling commands you can use for monitoring training. See the "DASK DASHBOARD SETUP" section in `submit_daxg.sh` for this.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub. Note that this is adapted from code used in the paper "Machine learning in Alzheimer's Genetics". To ensure nothing related to private genetic data or files was transferred to a public repo it was rewritten. Flagging issues is very welcome as they likely just got lost in the transfer.

## License

This project is licensed under the MIT License.