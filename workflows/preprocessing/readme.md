# Preprocessing workflow

This is a fork of the [plink2hdf5](https://github.com/seafloor/plink2hdf5) workflow with additional QC steps added.

It requires it's own setup/installation to run on top of whatever was installed to run daxos.

# Installation
Download and install the requirements
```
git clone https://github.com/seafloor/daxos.git
cd daxos/workflows/preprocessing
```

## Install a base snakemake env for runninng the workflow
```
module load your/base/conda/module
conda config --set channel_priority strict
conda create -f requirements.yaml
conda activate snakemake
```

## Install the workflow env on the head node
Install the separate conda env that will be loaded on compute nodes to run the pipelines. Must be done now on the head node, as most clusters have no internet access on compute nodes:
```
cd workflow
conda create -f envs/python_env.yaml
```

Note that the download_plink rules from the snakefile will automatically be run on the head node as they are defined as local rules. To ensure this works, snakemake should always be called from the head node, and will then submit jobs for each of the rules (i.e. don't start an interactive job and then try to run the snakemake workflow from a compute node).

## Customise the config file
Everything is managed via a config file - you shouldn't need to touch the snakefile except to add more QC steps.
See config.yaml or edit a clean copy from example_config.yaml.

```
cp example_config.yaml config.yaml
vim config.yaml # add your own paths etc.
```

## Running

### Check a dry run to see the workflow:
```
snakemake -np --cores 1 --jobs 1
```

### Run the full workflow on slurm

Make sure config.yaml exists and is correct first!

```
snakemake -p --cores 1 --executor slurm --workflow-profile profiles/slurm --jobs 1
```

## Debugging possible conda fixes
Some systems have issues with the conda version not being found by snakemake. You make need to run the lines below to fix this.
```
~/.conda/envs/snakemake/bin/conda init
source ~/.bashrc
```

## License and contributions
See the root daxos readme for these.
