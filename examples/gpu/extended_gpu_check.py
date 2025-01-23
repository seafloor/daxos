"""
Example of training with Dask on GPU
Modified from XGB docs to add timing, from cudf and suitable for xgb 1.7.6
====================================
"""

import cupy as cp
# import dask_cudf
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client
import dask
from dask_cuda import LocalCUDACluster
import time

from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix


def using_dask_matrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
    start_time = time.time()

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local DMatrix scatter
    # around workers.
    dtrain = DaskDMatrix(client, X, y)

    # Use train method from xgboost.dask instead of xgboost.  This distributed version
    # of train returns a dictionary containing the resulting booster and evaluation
    # history obtained from evaluation metrics.
    output = dxgb.train(
        client,
        {
            "verbosity": 3,  # Detailed logging
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )
    elapsed_time = time.time() - start_time
    print(f"\n### Training with DMatrix completed in {elapsed_time:.2f} seconds ###")

    bst = output["booster"]
    history = output["history"]

    # you can pass output directly into `predict` too.
    prediction = dxgb.predict(client, bst, dtrain)
    print("Evaluation history:", history)
    return prediction


def using_quantile_dask_dmatrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
    start_time = time.time()

    """`DaskQuantileDMatrix` is a data type specialized for `hist` tree methods for
     reducing memory usage.

    .. versionadded:: 1.2.0

    """
    X = dd.from_dask_array(X)
    y = dd.from_dask_array(y)

    # `DaskQuantileDMatrix` is used instead of `DaskDMatrix`, be careful that it can not
    # be used for anything else other than training unless a reference is specified. See
    # the `ref` argument of `DaskQuantileDMatrix`.
    dtrain = dxgb.DaskQuantileDMatrix(client, X, y)
    output = dxgb.train(
        client,
        {
            "verbosity": 3,  # Detailed logging
            "tree_method": "gpu_hist",
        },
        dtrain,
        num_boost_round=4,
    )
    elapsed_time = time.time() - start_time
    print(f"\n### Training with DaskQuantileDMatrix completed in {elapsed_time:.2f} seconds ###")

    prediction = dxgb.predict(client, output, X)
    return prediction


if __name__ == "__main__":
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker process.
    print('\n--> Initialising cluster')
    with LocalCUDACluster(n_workers=1, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            # generate some random data for demonstration
            print('\n--> Simulating data')
            with dask.config.set({"array.random.seed": 42}):
                m = 100000
                n = 100
                X = da.random.normal(size=(m, n))
                y = X.sum(axis=1)

            print("\n--> Running Using DMatrix")
            from_dmatrix = using_dask_matrix(client, X, y)

            print("\n--> Running Using DaskQuantileDMatrix")
            from_ddqdm = using_quantile_dask_dmatrix(client, X, y)
