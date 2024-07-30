import pytest
from unittest.mock import MagicMock, patch
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster
from daxos.distribute import spin_cluster, scale_cluster
import re

def get_threads_from_worker(worker):
    # Extract the string representation of the worker
    worker_str = str(worker)
    # Use regex to find the number of threads
    match = re.search(r'threads: (\d+)', worker_str)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Number of threads not found in worker string")

def test_spin_cluster_local():
    cluster = spin_cluster('local', n_threads=4, local_dir='/tmp')
    assert isinstance(cluster, LocalCluster)
    assert len(cluster.workers.keys()) == 1
    assert get_threads_from_worker(cluster.workers[0]) == 4

@patch('daxos.distribute.SLURMCluster')
def test_spin_cluster_distributed(mock_slurmcluster):
    cluster = spin_cluster('distributed', n_threads=4, local_dir='/tmp', 
                           mem='10G', walltime='01:00:00', interface='lo0')
    mock_slurmcluster.assert_called_once_with(cores=4, processes=1, memory='10G',
                                              walltime='01:00:00', local_directory='/tmp',
                                              interface='lo0', queue=None)
    assert isinstance(cluster, MagicMock)  # SLURMCluster is mocked

def test_spin_cluster_invalid():
    with pytest.raises(ValueError, match='Cluster type not recognised'):
        spin_cluster('invalid_type', n_threads=4, local_dir='/tmp')

@patch('daxos.distribute.SLURMCluster')
def test_scale_cluster_distributed(mock_slurmcluster):
    cluster = mock_slurmcluster.return_value
    scale_cluster(cluster, 'distributed', n_worker_arg=2, n_thread_per_worker_arg=4, mem_per_worker_arg='10G')
    cluster.scale.assert_called_once_with(jobs=2)

def test_scale_cluster_local():
    cluster = LocalCluster(n_workers=1, threads_per_worker=4, local_directory='/tmp')
    scale_cluster(cluster, 'local', n_worker_arg=2, n_thread_per_worker_arg=4, mem_per_worker_arg='10G')
    # Ensure that no scaling occurs for LocalCluster
    assert len(cluster.workers.keys()) == 1

def test_scale_cluster_invalid():
    with pytest.raises(ValueError, match='Cluster type not recognised'):
        scale_cluster(None, 'invalid_type', n_worker_arg=2, n_thread_per_worker_arg=4, mem_per_worker_arg='10G')

if __name__ == "__main__":
    pytest.main()
