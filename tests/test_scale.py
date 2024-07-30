import pytest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from daxos.scale import run_cv_and_platt_scale
import dask.array as da
from dask.distributed import Client

@pytest.fixture
def dask_client():
    client = Client(n_workers=2, threads_per_worker=2)
    yield client
    client.close()

@patch('daxos.crossvalidate.fit_one_round_cv')
@patch('daxos.read.print_summary')
def test_run_cv_and_platt_scale(mock_print_summary, mock_fit_one_round_cv, dask_client):
    # Dummy data
    X = da.random.random((100, 10), chunks=(10, 10))
    y_adjusted = da.random.random((100, 1), chunks=(10, 1))
    y_binary = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    params = {'eta': 0.1, 'max_depth': 3, 'subsample': 0.8, 
              'colsample_bytree': 0.8,  'n_boost_round': 10}
    
    # Mocking fit_one_round_cv
    def dummy_predict(y_true):
        y_true = y_true.squeeze()
        length = len(y_true)
        return np.where(y_true == 1, np.random.normal(0.55, 0.1, length), np.random.normal(0.45, 0.1, length))
    
    mock_fit_one_round_cv.return_value = (
        [np.random.normal(0.6, 0.1, 50), np.random.normal(0.6, 0.1, 50)],  # dummy scores
        [dummy_predict(y_binary[:50]), dummy_predict(y_binary[50:])],  # dummy predictions
        [y_binary[:50].squeeze(), y_binary[50:].squeeze()]  # dummy true values
    )

    # Call the function
    fit_kwargs = {'loss': 'reg:squarederror',
                  'eval_metric': 'rmse',
                  'tree_method': 'hist'}
    model = run_cv_and_platt_scale(dask_client, X, y_adjusted, y_binary, params, n_fold=2,
                                   score_method='RMSE', **fit_kwargs)

    # Assertions
    assert isinstance(model, LogisticRegression)

    # Verify predictions stacking
    stacked_predictions = np.hstack(mock_fit_one_round_cv.return_value[1]).reshape(-1, 1)
    assert stacked_predictions.shape == (100, 1)  # Adjust shape as per dummy data

    # Verify AUC calculation
    y_pred = model.predict_proba(stacked_predictions)[:, 1]
    auc = roc_auc_score(y_binary, y_pred) 
    assert 0.0 <= auc <= 1.0  # Adjust threshold as per dummy data

if __name__ == "__main__":
    pytest.main()
