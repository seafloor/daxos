import pytest
import dask.array as da
import xgboost as xgb
import numpy as np
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from dask.distributed import Client
from daxos.explain import (subset_after_shap, subset_predictors, subset_predictors_and_refit,
                           collect_importances)

@pytest.fixture
def sample_data():
    np.random.seed(0)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=(100, 1))
    colnames = [f'feature_{i}' for i in range(10)]
    used_cols = colnames[:5]
    return X, y, colnames, used_cols

@pytest.fixture
def dask_client():
    client = Client(n_workers=2, threads_per_worker=2)
    yield client
    client.close()

def test_subset_after_shap(sample_data):
    X, y, colnames, used_cols = sample_data
    shap_values = np.random.rand(100, len(colnames) + 1)  # Adding bias term
    result = subset_after_shap(used_cols, colnames, shap_values,
                               value_type='marginal', drop_bias=False)
    assert result.shape[1] == len(used_cols) + 1  # +1 for bias term

    result = subset_after_shap(used_cols, colnames, shap_values,
                               value_type='marginal', drop_bias=True)
    assert result.shape[1] == len(used_cols)  # without bias term

def test_subset_predictors(sample_data):
    X, y, colnames, used_cols = sample_data
    X_da = da.from_array(X, chunks=(10, 10))
    X_subset, out_cols = subset_predictors(X_da, colnames, used_cols)
    assert X_subset.shape[1] == len(used_cols)  

@patch('daxos.explain.persist_daskdmatrix')
@patch('daxos.explain.fit_dask_xgb')
def test_subset_predictors_and_refit(mock_fit_dask_xgb, mock_persist_daskdmatrix, sample_data, dask_client):
    X, y, colnames, used_cols = sample_data
    X_da = da.from_array(X, chunks=(10, 10))
    y_da = da.from_array(y, chunks=(10, 1))
    params = {'objective': 'binary:logistic', 'max_depth': 3, 'eta': 0.1}

    mock_persist_daskdmatrix.return_value = MagicMock()
    mock_fit_dask_xgb.return_value = (MagicMock(), MagicMock())

    dtrain, bst, out_cols = subset_predictors_and_refit(dask_client, X_da, y_da,
                                                        used_cols, colnames, params)
    
    assert len(out_cols) == len(used_cols)
    mock_persist_daskdmatrix.assert_called_once()
    mock_fit_dask_xgb.assert_called_once()

@patch('daxos.explain.pd.DataFrame.to_csv')
@patch('daxos.explain.da.to_zarr')
@patch('daxos.explain.xgb.dask.predict')
def test_collect_importances(mock_predict, mock_to_zarr, mock_to_csv, sample_data, dask_client, tmp_path):
    X, y, colnames, _ = sample_data
    bst = MagicMock()
    dtrain = MagicMock()
    mock_predict.return_value = da.from_array(np.random.rand(100, 10 + 1), chunks=(10, 11))  # Adding bias term

    out_dir = tmp_path
    out_prefix = 'test'
    run_shap_main = True
    run_shap_inter = False

    importance = collect_importances(dask_client, bst, dtrain, colnames, out_dir, out_prefix, run_shap_main, run_shap_inter)
    
    assert isinstance(importance, pd.DataFrame)
    mock_predict.assert_called_once()
    mock_to_zarr.assert_called_once()
    mock_to_csv.assert_called()

if __name__ == "__main__":
    pytest.main()
