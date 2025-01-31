from dask.distributed import LocalCluster
from dask_cuda import LocalCUDACluster
from dask_jobqueue import SLURMCluster
import pathlib
import os


def spin_cluster(cluster_type, n_threads, local_dir, processes=1, mem='10G', walltime='01:00:00', interface='ib0',
                 queue=None, logdir=None, gpu=True, gpu_resources='gpu:1', worker_prologue=None):
    print("Parsing cluster arguments...")
    
    if worker_prologue is None:
        print('No modules passed to load on worker')
        worker_prologue = []
    else:
        worker_prologue = [
            f"module load {worker_prologue['gcc_module']}",
            f"module load {worker_prologue['cuda_module']}",
            f"module load {worker_prologue['conda_module']}",
            f"conda activate {worker_prologue['conda_env']}"
        ]

        if gpu:
            worker_prologue.extend(
                [
                    'echo -e "\\n################################################################################"',
                    'echo -e "####################### Checking GPUs Before Grabbing ##########################"',
                    'echo "Initial CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"',
                    "unset CUDA_VISIBLE_DEVICES",
                    'echo "CUDA_VISIBLE_DEVICES after unset: $CUDA_VISIBLE_DEVICES"',
                    "nvidia-smi",
                    'echo -e "################################################################################"',
                    'echo -e "################################################################################\\n"',
                    'echo -e "\\n################################################################################"',
                    'echo -e "######################## Checking GPUs After Grabbing ##########################"',
                    "python scripts/grab_gpus.py",
                    'echo "CUDA_VISIBLE_DEVICES after grabbing GPUs: $CUDA_VISIBLE_DEVICES"',
                    'echo -e "################################################################################"',
                    'echo -e "################################################################################\\n"',
                ]
            )
        
        print(f"Worker prologue passed as: {worker_prologue}")

    walltime = walltime.replace('_', ':')

    if cluster_type == 'local':
        print(f"\n--> Starting LocalCluster with 1 worker and {n_threads} threads on {'GPU' if gpu else 'CPU'}")
        if gpu:
            return LocalCUDACluster(
                n_workers=1,
                threads_per_worker=n_threads,
                local_directory=local_dir,
                device_memory_limit="7.5GB")
        else:
            return LocalCluster(
                n_workers=1, 
                threads_per_worker=n_threads, 
                local_directory=local_dir)
    elif cluster_type == 'distributed':
        print(f"\n--> Starting SLURMCluster using {interface} interface "
              f"with {n_threads} threads, {mem} mem and {walltime} walltime "
              f"on {'GPU' if gpu else 'CPU'}")
        
        job_extra = []
        worker_resources = []
        if gpu:
            print(f'Requesting GPU resources as {gpu_resources}')
            job_extra.append(f"--gres={gpu_resources}")
            worker_resources.append("--resources GPU=1")
            worker_resources.append("--memory-limit 7.5GB")

        return SLURMCluster(
                cores=n_threads,
                processes=processes,
                memory=mem,
                walltime=walltime,
                local_directory=local_dir,
                interface=interface,
                queue=queue,
                job_extra_directives=job_extra, # for extra sbatch requests like --gres
                job_script_prologue=worker_prologue, # for loading required modules etc. on workers
                worker_extra_args=worker_resources,  # for notifying dask scheduler of resources on workers
            )
    else:
        raise ValueError('Cluster type not recognised. Must be local or distributed.')


def scale_cluster(cluster, cluster_arg, n_worker_arg, n_thread_per_worker_arg, mem_per_worker_arg):
    if cluster_arg == 'distributed':
        total_compute = int(n_thread_per_worker_arg * n_worker_arg)
        total_memory = int(int(mem_per_worker_arg.replace("G", "")) * n_worker_arg)
        print(f'Scaling up cluster to {n_worker_arg} workers')
        print(f'Requesting total compute of {total_compute} threads and {total_memory}GB mem')
        print('!Note: requested memory not necessarily allocated - check sacct/seff')

        cluster.scale(jobs=n_worker_arg)
    elif cluster_arg == 'local':
        print('Local cluster used: not scaled above single worker')
    else:
        raise ValueError('Cluster type not recognised. Must be local or distributed.')
