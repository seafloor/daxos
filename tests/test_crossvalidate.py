import pytest
import dask.array as da
import numpy as np
import pandas as pd
import xgboost as xgb
from dask.distributed import Client
from daxos.crossvalidate import (map_partitions_to_workers, map_x_y_partitions_to_workers, 
                                  persist_daskdmatrix, set_random_search_distributions, 
                                  flatten_if_2d, score_model, fit_dask_xgb, 
                                  xgb_dask_cv, get_best_cv, chunked_train_test_split, 
                                  chunked_kfold_split)

@pytest.fixture
def dask_client():
    client = Client(n_workers=2, threads_per_worker=2)
    yield client
    client.close()

def test_map_partitions_to_workers():
    workers = ['worker1', 'worker2']
    partitions = ['partition1', 'partition2', 'partition3', 'partition4']
    splits = [[0, 1], [2, 3]]
    mapping = map_partitions_to_workers(workers, partitions, splits)
    assert mapping == {'partition1': 'worker1', 'partition2': 'worker1', 
                       'partition3': 'worker2', 'partition4': 'worker2'}

def test_map_x_y_partitions_to_workers(dask_client):
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    mapping = map_x_y_partitions_to_workers(dask_client, X, y)
    assert isinstance(mapping, dict)
    assert len(mapping) == X.npartitions + y.npartitions

def test_persist_daskdmatrix(dask_client):
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    dtrain = persist_daskdmatrix(dask_client, X, y)
    assert isinstance(dtrain, xgb.dask.DaskDMatrix)

def test_set_random_search_distributions():
    param_sampler = set_random_search_distributions(n_iter=5)
    params = list(param_sampler)
    assert len(params) == 5

def test_flatten_if_2d():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    flattened = flatten_if_2d(a)
    assert flattened.ndim == 1

def test_score_model():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    auc_score = score_model(y_true, y_pred, score_method='AUC')
    assert 0 <= auc_score <= 1

def test_fit_dask_xgb(dask_client):
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    dtrain = persist_daskdmatrix(dask_client, X, y)
    params = {'eta': 0.1, 'max_depth': 3, 'subsample': 0.8, 
              'colsample_bytree': 0.8,  'n_boost_round': 10}
    bst, history = fit_dask_xgb(dask_client, dtrain, params)
    assert isinstance(bst, xgb.Booster)
    assert 'train' in history

def test_incremental_fit_xgb(dask_client):
    pass

def test_xgb_dask_cv(dask_client):
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    params = {'eta': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}
    cv_results, cv_y_pred = xgb_dask_cv(dask_client, X, y, params,
                                        boost_rounds=10)
    assert isinstance(cv_results, pd.DataFrame)
    assert isinstance(cv_y_pred, pd.DataFrame)

def test_get_best_cv():
    cv_scores = pd.DataFrame({'eta': [0.1, 0.2], 'max_depth': [3, 3], 'subsample': [0.8, 0.9], 
                              'colsample_bytree': [0.8, 0.7], 'score': [0.9, 0.8]})
    best_params, _ = get_best_cv(cv_scores)
    assert isinstance(best_params, dict)
    assert best_params['eta'] == 0.1

def test_chunked_train_test_split():
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    X_train, X_test, *_ = chunked_train_test_split(X, y, row_chunks=10, train_size=0.8)
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20

def test_chunked_kfold_split():
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    kfolds = chunked_kfold_split(X, y, n_splits=5)
    assert len(kfolds) == 5

def test_cv_xgb(dask_client):
    pass

if __name__ == "__main__":
    pytest.main()
