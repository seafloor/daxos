import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm import tqdm
import dask.array as da
import numpy as np


def read_covars(f, verbose=True):
    print('\n--> Reading covar file assuming plink format')
    cov = pd.read_csv(f, sep='\s+')

    if verbose:
        print('Head of covar file:\n')
        print(cov.head())

    return cov


def align_covars_to_rows(cov, ids):
    print(f'\nAligning covariate file of shape {cov.shape} to ids in X')
    cov = cov.set_index('IID').loc[ids, :].reset_index()
    print(f'{cov.shape} rows present after alignment')
    print('\nHead of aligned covar file:\n')
    print(cov.head())

    return cov


def parse_covars(f, ids):
    cov = read_covars(f)
    cov = align_covars_to_rows(cov, ids)

    cov = cov.drop(columns=['FID', 'IID'], errors='ignore')
    print(f'Covariates to be used for deconfounding: {cov.columns.to_numpy()}\n')

    return cov.to_numpy(dtype=np.float32)


def run_regressions(X, covars):
    betas = []

    for column in tqdm(np.arange(X.shape[1])):
        lr = LinearRegression()

        # drop rows in covariates and X if missing values for predictor in X
        not_nan_bool = ~np.isnan(X[:, column])
        lr.fit(covars[not_nan_bool, :], X[not_nan_bool, column])

        betas.append(lr.coef_.reshape(-1, 1))

    return np.hstack(betas).astype(np.float32)


def calculate_betas_for_y(y, covars, method='logistic'):
    if method == 'linear':
        betas = LinearRegression().fit(covars, y).coef_.reshape(-1, 1)
    elif method == 'logistic':
        betas = LogisticRegression().fit(covars, y).coef_.reshape(-1, 1)
    else:
        raise ValueError(f'Method {method} for deconfounding y not recognised.')

    return betas


def calculate_residuals_for_y(y, cov, betas):
    cov_dot_beta = np.dot(cov, betas)  # pulled into separate line for sanity checking dimensions

    assert cov_dot_beta.shape == y.shape, \
        f'Dot product of covariates and betas needs shape {y.shape} but is shape {cov_dot_beta.shape}'

    residuals = np.subtract(y, cov_dot_beta)

    return residuals.astype(np.float32)


def calculate_betas_for_x(X, covars):
    betas = run_regressions(X, covars)

    betas_shape_expected = (covars.shape[1], X.shape[1])

    assert betas.shape == betas_shape_expected, f'betas have shape {betas.shape} but should be {betas_shape_expected}'
    print(f'n_covariates x n_features matrix of shape {betas.shape} created')

    return betas


def calculate_residuals_for_x(X, covars, betas, chunk_size=1000):
    X = da.from_array(X, chunks=chunk_size)
    covars = da.from_array(covars, chunks=chunk_size)
    betas = da.from_array(betas, chunks=chunk_size)

    cov_dot_beta = da.dot(covars, betas)

    assert cov_dot_beta.shape == X.shape, \
        f'Dot product of covariates and betas needs shape {X.shape} but is shape {cov_dot_beta.shape}'

    residuals = X - cov_dot_beta
    residuals = residuals.compute()
    assert X.shape == residuals.shape, \
        f'Raw and transformed X have different shapes: {X.shape} vs {residuals.shape}'

    return residuals.astype(np.float32)
