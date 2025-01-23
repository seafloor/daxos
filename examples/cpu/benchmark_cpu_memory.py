from memory_profiler import memory_usage
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster
import time
import h5py
import sys
import os
from tabulate import tabulate
import xgboost as xgb
import logging
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxos.read import read_ml


def measure_memory_and_time(func, *args, **kwargs):
    """
    Measure peak memory usage and execution time for a function.
    """
    start_time = time.time()
    mem_usage = memory_usage((func, args, kwargs), interval=0.1, retval=True)
    elapsed_time = time.time() - start_time
    peak_memory = max(mem_usage[0])  # memory_usage returns a tuple: (list of memory values, return value)
    result = mem_usage[1]
    return result, elapsed_time, peak_memory


def using_quantile_dask_dmatrix(client: Client, X: da.Array, y: da.Array) -> tuple:
    """
    Train a model and measure time and memory usage.
    """
    def train():
        X_dask = dd.from_dask_array(X)
        y_dask = dd.from_dask_array(y)

        dtrain = xgb.dask.DaskQuantileDMatrix(client, X_dask, y_dask)
        xgb.dask.train(
            client,
            {
                "verbosity": 3,
                "tree_method": "hist",
            },
            dtrain,
            num_boost_round=50,
        )

    # Measure memory and time
    _, elapsed_time, peak_memory = measure_memory_and_time(train)
    return elapsed_time, peak_memory


def process_file(client: Client, file_path: str) -> tuple:
    file_name = os.path.basename(file_path)
    print(f"\n--> Processing {file_name}")

    try:
        with h5py.File(file_path, 'r') as f:
            X, y, rows, columns = read_ml(file_path, f, row_chunks=100, x_key='x', y_key='y')

            # Convert datasets to Dask arrays while the file is still open
            if not isinstance(X, da.Array):
                X = da.from_array(X, chunks=(100, None))
            if not isinstance(y, da.Array):
                y = da.from_array(y, chunks=(100, None))

            # Call the training function while the dataset is still open
            time_taken, peak_memory = using_quantile_dask_dmatrix(client, X, y)

        # Format the results
        time_str = f"{time_taken:.2f}" if time_taken is not None else "Error"
        memory_str = f"{peak_memory:.2f} MB" if peak_memory is not None else "Error"
        return file_name, time_str, memory_str
    except Exception as e:
        # Catch and report errors in both timing and memory
        return file_name, "Error", str(e)


if __name__ == "__main__":
    # Updated file paths
    file_paths = [
        '../data/example_data_train.hdf5',
        '../data/large_example_data_p100_train.hdf5',
        '../data/large_example_data_p1000_train.hdf5'
    ]

    results = []

    print(
        """
        ################################################
        ################################################
        #           Benchmarking CPU-daxos             #
        ################################################
        ################################################
        """
    )

    print('\n--> Initialising cluster')
    with LocalCluster(n_workers=1, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            for file_path in file_paths:
                results.append(process_file(client, file_path))

    # Print summary table
    print("\n=== Summary ===")
    headers = ["File", "Time (seconds)", "Peak Memory Usage (MB)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
