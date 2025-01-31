from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import time

sleep_for = 10

# GPU toggle
gpu = True  # Set to False to disable GPU-specific settings

# Define modules and environment settings
worker_prologue = {
    "gcc_module": "gcc/11.3.1/compilers",
    "cuda_module": "nvidia/cuda/12.0/compilers",
    "conda_module": "conda/23.11-py311",
    "conda_env": "daxos"
}

# Base job script prologue
job_script_prologue = [
    f"module load {worker_prologue['gcc_module']}",
    f"module load {worker_prologue['cuda_module']}",
    f"module load {worker_prologue['conda_module']}",
    f"conda activate {worker_prologue['conda_env']}"
]

# Add GPU-specific commands if GPU is enabled
if gpu:
    job_script_prologue.extend([
        'echo -e "\\n################################################################################"',
        'echo -e "############################# Verifying GPU support ##############################"',
        "nvidia-smi",
        'echo -e "################################################################################"',
        'echo -e "################################################################################\\n"'
    ])

# Define the SLURMCluster with GPU-specific args if needed
worker_extra_args = ["--gres=gpu:1"] if gpu else []

cluster = SLURMCluster(
    cores=1,
    memory="1GB",
    processes=1,
    walltime="00:05:00",
    job_script_prologue=job_script_prologue,
    worker_extra_args=worker_extra_args,
)

# Scale the cluster to 1 worker
cluster.scale(jobs=1)

# Connect to the cluster
client = Client(cluster)

# Verify the cluster is working by printing cluster information
print(cluster)
print("Dask client created successfully!")
print(f"Sleeping for {sleep_for} second to allow time to manually check job queues...")
time.sleep(sleep_for)

# Shut down the cluster after testing
print("Shutting down cluster")
client.close()
cluster.close()
