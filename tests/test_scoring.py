import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from daxos.scoring import fit_rf, fit_ols, auc

@pytest.fixture
def sample_data():
    np.random.seed(0)
    data = pd.DataFrame({
        'IID': np.arange(100),
        'y_true': np.random.randint(0, 2, size=100),
        'y_pred': np.random.rand(100),
        'covar1': np.random.rand(100),
        'covar2': np.random.rand(100)
    })
    return data

def test_fit_rf(sample_data):
    covars = ['covar1', 'covar2']
    with patch('daxos.scoring.RandomizedSearchCV') as mock_cv:
        mock_cv_instance = mock_cv.return_value
        mock_cv_instance.best_params_ = {
            'max_depth': 5,
            'min_samples_split': 2,
            'max_features': 3
        }
        mock_cv_instance.best_score_ = -1.0
        mock_cv_instance.fit.return_value = None

        fv, res = fit_rf(sample_data, covars, rf_trees=5, iter=3, nsubsample=50)
        
        assert len(fv) == 100
        assert len(res) == 100

def test_fit_ols(sample_data):
    covars = ['covar1', 'covar2']
    fv, res = fit_ols(sample_data, covars)
    
    assert len(fv) == 100
    assert len(res) == 100

def test_auc_no_covars(sample_data):
    a, r, fv, res = auc(sample_data, y_true='y_true', y_pred='y_pred', covars=None)
    assert isinstance(a, float)
    assert r is None
    assert fv is None
    assert res is None

def test_auc_with_covars_ols(sample_data):
    covars = ['covar1', 'covar2']
    a, r, fv, res = auc(sample_data, y_true='y_true', y_pred='y_pred', covars=covars, model='ols')
    
    assert isinstance(a, float)
    assert isinstance(r, float)
    assert len(fv) == 100
    assert len(res) == 100

def test_auc_with_covars_rf(sample_data):
    covars = ['covar1', 'covar2']
    with patch('daxos.scoring.RandomizedSearchCV') as mock_cv:
        mock_cv_instance = mock_cv.return_value
        mock_cv_instance.best_params_ = {
            'max_depth': 5,
            'min_samples_split': 2,
            'max_features': 3
        }
        mock_cv_instance.best_score_ = -1.0
        mock_cv_instance.fit.return_value = None
        rf_kwargs = {'rf_trees': 5, 'iter': 3, 'nsubsample': 50}
        a, r, fv, res = auc(sample_data, y_true='y_true', y_pred='y_pred', covars=covars,
                            model='rf', **rf_kwargs)
        
        assert isinstance(a, float)
        assert isinstance(r, float)
        assert len(fv) == 100
        assert len(res) == 100

if __name__ == "__main__":
    pytest.main()
