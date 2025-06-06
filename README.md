# DAXOS: Dask and XGBoost on SNPs

[![DOI](https://zenodo.org/badge/577098629.svg)](https://doi.org/10.5281/zenodo.15526395)

DAXOS is a Python library designed to efficiently run Dask and XGBoost on SNP data, particularly for use on high-performance computing (HPC) systems that use a SLURM management system. The code is optimized to reduce memory usage and is tailored for handling genetic data from PLINK files.

## Features

- **Memory-efficient processing:** Utilizes Dask to handle large datasets with reduced memory usage. This includes *lots* of optimisations like pre-shuffling and chunking for dask, and custom load balancing, as by default dask is unaware of how much memory XGB will use and loads too many chunks on a given worker, using in-place prediction where optimal, a custom CV function to minimise data movement to/from workers
- **XGBoost integration:** Leverages the power of XGBoost for fast and scalable machine learning. All the core work is done using XGB on dask arrays.
- **HPC optimization:** Designed specifically for HPC environments. If you're working on AWS or a personal computer then there are probably better workflows. If you're on SLURM HPC system with moderate memory per node then daxos should be helpful.
- **PLINK to HDF5 conversion:** Includes convenience functions to convert PLINK files to HDF5 format. This includes making sure the ref allele is the same across train/test splits, low-memory pre-shuffling and processing with numpy.
- **Covariate adjustment:** Functions for adjusting covariates in genetic data before and after analysis. This includes genome-wide adjustment as has been advised for random forests (1), and a post-modelling adjustment for the non-linear effect of confounders.
- **Cross-validation, refitting and prediction:** Support for model cross-validation, refitting, and prediction, and explanation through SHAP values. The latter uses the SHAP "prediction" built in to the python XGB API. Prediction in the test set can run Platt scaling (correctly implemented using the CV test chunk predictions).

## Installation

To install the required dependencies, use the provided `requirements.yaml` file with conda:

```sh
module load your/base/conda/module
conda config --set channel_priority strict
conda env create -f requirements.yaml
conda activate daxos
```

You will also need to make the package available in the python path, is it's not been installed:

```sh
cd your/install/dir/daxos
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## GPU support

*Note: GPU support is experimental. I recommend running the CPU version using the preprocessing and training snakemake workflows in /workflows for now. For this, please install as above and then head to the respective workflow directories to set up any specific conda environments.*

On an HPC cluster you likely don't have GPUs on the head node, so by default pip-installed xgboost will not have gpu support. This is easy to fix by building manually. The example below should be adapted to the modules available on your cluster. Note if you've already run the conda installation above then you'll need to uninstall with `conda remove xgboost`. You may need to also cleanup with `conda clean --all` and delete the python shared object for the CPU xgboost with `rm -f .conda/envs/daxos/lib/libxgboost.so` to avoid conflicts.

*On the head node:*

```
git clone --branch release_1.7.6 https://github.com/dmlc/xgboost.git
cd xgboost
git submodule update --init --recursive
```

*Access a GPU node, e.g.:*

```
salloc --time=01:00:00 --nodes=1 --ntasks=1 --tasks-per-node=1 --mem=20G --job-name=xgbinstall --gres=gpu:1 --cpus-per-task=1
srun --pty /bin/bash
```

*On the compute node with GPUs:*

```
module load gcc/11.3.1/compilers
module load nvidia/cuda/12.0/compilers

mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DBUILD_WITH_CUDA_CUB=ON
make -j$(nproc)
```

*Install the built package:*

```
module load conda/23.11-py311
conda activate daxos

cd ~/xgboost/python-package
python setup.py install

```

*Check version and GPU support*

```
# get version
python3 -c 'import xgboost as xgb; print(xgb.__version__)'

# basic GPU check
cd ~/daxos/examples/gpu
python3 minimal_gpu_check.py

# more detailed GPU check
cd ~/daxos/examples/gpu
python3 detailed_gpu_check.py
```

## Usage

Example SLURM submission scripts are included - please edit for your own file paths, module names etc. All submission scripts start with "submit", and often call python scipts which pull together all the functions in the main package.

### PLINK to HDF5 Conversion

Convert PLINK files to HDF5 format for more efficient processing using the `submit_plink_to_hdf5.sh` file. Note that you need plink installed, as well as the `shuf` unix tool (usually already available) on linux systems. To use the core functions outside of scripts, 

```python
# from /daxos
from daxos.read import read_raw_chunked
df = read_raw_chunked('tests/data/dummy_plink.raw', nrows=100)
```

### Covariate Adjustment

Adjust for covariates in your genetic data before analysis using the script `submit_adjust_hdf5_for_covars.sh`. After analysis, covariate adjusted predictions in the test set can be generated using daxos/scoring.py, which runs OLS or RF (preferred) adjustment for covariates. The latter is important to adjust for the non-linear effects of covariates when we're running non-linear prediction models. Again you can just access the core functions from the package too, e.g.

```python
# from /daxos
from daxos.deconfound import calculate_betas_for_x, calculate_residuals_for_x
from daxos.read import readml
hdf5_file = 'tests/data/dummy_data.hdf5'

with h5py.File(hdf5_file) as f:
    X, y, rows, columns = read.read_ml(hdf5_file, f, **kwargs)
    X = X.compute()
    y = y.compute()
    covars = your_covar_reading_func()
    
    x_betas = deconfound.calculate_betas_for_x(X, covars)
    X_residuals = deconfound.calculate_residuals_for_x(X, covars, x_betas)
```

### Training and predicting from models
The script `submit_daxg.sh` is supposed to be a helper script, but it's long and complicated for a slurm submission script. You can also just called the cv.py, refit.py or predict.py scripts with your own submission script.

The good thing it does have is something that prints the node and tunneling commands you can use for monitoring training. See the "DASK DASHBOARD SETUP" section in `submit_daxg.sh` for this.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub. Note that this is adapted from code used in the paper "Machine learning in Alzheimer's Genetics" (2). To ensure nothing related to private genetic data or files was transferred to a public repo it was rewritten. Flagging issues is very welcome as they likely just got lost in the transfer.

## License

This project is licensed under the MIT License.

## References
1. Zhao, Yang et al. (2012). “Correction for population stratification in random forest analysis”. International journal of epidemiology 41.6, pp. 1798–1806.
2. Bracher-Smith et al., in submission.
