# Preprocessing workflow

This is a minimal workflow for running the refitting and prediction parts of training for convenience. 

It requires it's own setup/installation to run on top of whatever was installed to run daxos. If you've already installed the snakemake environment as detailed in workflows/preprocessing and the main daxos environment then you can skip ahead to customising the config files.

# Installation
Download and install the requirements
```
git clone https://github.com/seafloor/daxos.git
cd daxos/workflows/training
```

## Install a base snakemake env for runninng the workflow
```
module load your/base/conda/module
conda config --set channel_priority strict
conda create -f requirements.yaml
conda activate snakemake
```

## Install the workflow env on the head node
For the training workflow, this is the daxos env detailed in the root readme. If you have that installed then skip ahead, else follow the steps in the root readme for installation.

## Customise the config file
Everything is managed via a config file - you shouldn't need to touch the snakefile except to add more steps.
See config_cpu.yaml and config_gpu.yaml. You will need to use your own paths for files and the names of modules to load on your hpc system.

## Running

### Check a dry run to see the workflow:
```
snakemake -np --cores 1 --jobs 1
```

### Run the full workflow on slurm

Make sure config.yaml exists and is correct for your data. You also need to choose between the cpu and gpu version for both the base config.yaml and the slurm job settings in profiles/slurm_cpu/config.yaml or profiles/slurm_gpu/config.yaml.

```
# CPU version
snakemake -p --cores 1 --executor slurm --configfile config_cpu.yaml --workflow-profile profiles/slurm_cpu --jobs 3

# GPU version
snakemake -p --cores 1 --executor slurm --configfile config_gpu.yaml --workflow-profile profiles/slurm_gpu --jobs 3
```

## Debugging possible conda fixes
Some systems have issues with the conda version not being found by snakemake. You make need to run the lines below to fix this.
```
~/.conda/envs/snakemake/bin/conda init
source ~/.bashrc
```

## License and contributions
See the root daxos readme for these.
