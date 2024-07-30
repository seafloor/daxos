import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from daxos.deconfound import (read_covars, align_covars_to_rows, parse_covars, run_regressions,
                              calculate_betas_for_y, calculate_residuals_for_y, calculate_betas_for_x,
                              calculate_residuals_for_x)
import dask.array as da

@pytest.fixture
def sample_covariate_file(tmp_path):
    df = pd.DataFrame({
        'FID': [1, 2, 3, 4, 5],
        'IID': [101, 102, 103, 104, 105],
        'covar1': np.random.rand(5),
        'covar2': np.random.rand(5)
    })
    file_path = tmp_path / 'covariates.txt'
    df.to_csv(file_path, sep=' ', index=False)
    return str(file_path)

@pytest.fixture
def sample_data():
    np.random.seed(0)
    X = np.random.rand(5, 2)
    y = np.random.choice([0, 1], size=(5, 1))
    covars = np.random.rand(5, 2)
    ids = np.array([101, 102, 103, 104, 105])
    return X, y, covars, ids

def test_read_covars(sample_covariate_file):
    covars = read_covars(sample_covariate_file)
    assert isinstance(covars, pd.DataFrame)
    assert covars.shape == (5, 4)

def test_align_covars_to_rows(sample_covariate_file, sample_data):
    _, _, _, ids = sample_data
    covars = read_covars(sample_covariate_file)
    aligned_covars = align_covars_to_rows(covars, ids)
    assert isinstance(aligned_covars, pd.DataFrame)
    assert aligned_covars.shape == (5, 4)

def test_parse_covars(sample_covariate_file, sample_data):
    _, _, _, ids = sample_data
    covars = parse_covars(sample_covariate_file, ids)
    assert isinstance(covars, np.ndarray)
    assert covars.shape == (5, 2)

def test_run_regressions(sample_data):
    X, _, covars, _ = sample_data
    betas = run_regressions(X, covars)
    assert isinstance(betas, np.ndarray)
    assert betas.shape == (2, 2)

def test_calculate_betas_for_y(sample_data):
    _, y, covars, _ = sample_data
    betas_linear = calculate_betas_for_y(y, covars, method='linear')
    betas_logistic = calculate_betas_for_y(y, covars, method='logistic')
    assert isinstance(betas_linear, np.ndarray)
    assert isinstance(betas_logistic, np.ndarray)
    assert betas_linear.shape == (2, 1)
    assert betas_logistic.shape == (2, 1)

def test_calculate_residuals_for_y(sample_data):
    _, y, covars, _ = sample_data
    betas = calculate_betas_for_y(y, covars, method='linear')
    residuals = calculate_residuals_for_y(y, covars, betas)
    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == y.shape

def test_calculate_betas_for_x(sample_data):
    X, _, covars, _ = sample_data
    betas = calculate_betas_for_x(X, covars)
    assert isinstance(betas, np.ndarray)
    assert betas.shape == (2, 2)

def test_calculate_residuals_for_x(sample_data):
    X, _, covars, _ = sample_data
    betas = calculate_betas_for_x(X, covars)
    residuals = calculate_residuals_for_x(X, covars, betas, chunk_size=2)
    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == X.shape

if __name__ == "__main__":
    pytest.main()
