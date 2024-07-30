from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


def spin_cluster(cluster_type, n_threads, local_dir, processes=1, mem='10G', walltime='01:00:00', interface='ib0',
                 queue=None):
    walltime = walltime.replace('_', ':')

    if cluster_type == 'local':
        print('\n--> Starting LocalCluster with 1 worker and {} threads'.format(n_threads))
        return LocalCluster(n_workers=1, threads_per_worker=n_threads, local_directory=local_dir)
    elif cluster_type == 'distributed':
        print(f'\n--> Starting SLURMCluster using {interface} interface '
              f'with {n_threads} threads, {mem} mem and {walltime} walltime')
        return SLURMCluster(cores=n_threads, processes=processes, memory=mem, walltime=walltime,
                            local_directory=local_dir, interface=interface, queue=queue)
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
