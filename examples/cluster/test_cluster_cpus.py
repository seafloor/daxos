from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import time

sleep_for = 10

# Define the SLURMCluster with job_script_prologue
cluster = SLURMCluster(
    cores=1,
    memory="1GB",
    processes=1,
    walltime="00:05:00",
    job_script_prologue=[
        "module load gcc/11.3.1/compilers",
        "module load nvidia/cuda/12.0/compilers",
        "module load conda/23.11-py311",
        "conda activate daxos",
        "echo 'Python version on worker:'",
        "python --version"
    ],
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

