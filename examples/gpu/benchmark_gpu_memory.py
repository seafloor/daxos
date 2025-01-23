import cupy as cp
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client
import dask
from dask_cuda import LocalCUDACluster
import time
import h5py
import sys
import os
from tabulate import tabulate
import xgboost as xgb
import logging
import re
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxos.read import read_ml


def using_quantile_dask_dmatrix(client: Client, X: da.Array, y: da.Array, file_name: str) -> float:
    start_time = time.time()

    logger = logging.getLogger("xgboost")
    logger.setLevel(logging.DEBUG)

    try:
        X = dd.from_dask_array(X)
        y = dd.from_dask_array(y)

        dtrain = xgb.dask.DaskQuantileDMatrix(client, X, y)

        xgb.dask.train(
            client,
            {
                "verbosity": 3,
                "tree_method": "gpu_hist",
            },
            dtrain,
            num_boost_round=50,
        )
    finally:
        pass  # Cleanup, if necessary

    elapsed_time = time.time() - start_time
    return elapsed_time


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
            time_taken = using_quantile_dask_dmatrix(client, X, y, file_name)

        return file_name, f"{time_taken:.2f}"
    except Exception as e:
        return file_name, "Error"


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
        #           Benchmarking GPU-daxos             #
        ################################################
        ################################################
        """
    )

    print('\n--> Initialising cluster')
    with LocalCUDACluster(n_workers=1, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            for file_path in file_paths:
                results.append(process_file(client, file_path))

    # Print summary table
    print("\n=== Summary ===")
    headers = ["File", "Time (seconds)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    print("\nNote: Refer to XGBoost logs above for peak GPU memory usage.")
