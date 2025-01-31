# starter script for worker nodes
# not in use at the moment
import os
import subprocess
import yaml

# Load configuration from the YAML file
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Setup function for Dask workers
def dask_setup_worker():
    config_path = "../config.yaml"  # Relative path to the config file
    config = load_config(config_path)

    # Extract module names from the config
    compiler_module = config['env']['compiler_module']
    cuda_module = config['env']['cuda_module']
    conda_module = config['env']['conda_module']
    conda_env = config['env']['job_env']

    # Load necessary modules
    print("\n--> Loading required modules for the worker")
    subprocess.run(["module", "load", compiler_module], check=True)
    subprocess.run(["module", "load", cuda_module], check=True)
    subprocess.run(["module", "load", conda_module], check=True)
    subprocess.run(["conda", "activate", conda_env], shell=True, check=True)
    print(f'Running {python3 --version}')

if __name__ == '__main__':
    dask_setup_worker()
