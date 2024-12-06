from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from split_hdf5 import read_ml
from split_ids import read_covars, check_covars
import argparse
import numpy as np
import h5py
import dask.array as da
import sys
import os

def align_covars_to_rows(cov, ids):
    print(f'\nAligning covariate file of shape {cov.shape} to ids in X')
    cov = cov.set_index('IID').loc[ids, :].reset_index()
    print(f'{cov.shape} rows present after alignment')
    print('\nHead of aligned covar file:\n')
    print(cov.head())

    return cov


def parse_covars(f, ids):
    cov = read_covars(f)
    check_covars(cov)

    cov = align_covars_to_rows(cov, ids)

    cov = cov.drop(columns=['FID', 'IID'], errors='ignore')
    print(f'Covariates to be used for deconfounding: {cov.columns.to_numpy()}\n')

    return cov.to_numpy(dtype=np.float32)


def run_regressions(X, covars):
    betas = []

    for column in np.arange(X.shape[1]):
        lr = LinearRegression()

        # drop rows in covariates and X if missing values for predictor in X
        not_nan_bool = ~np.isnan(X[:, column])
        lr.fit(covars[not_nan_bool, :], X[not_nan_bool, column])

        betas.append(lr.coef_.reshape(-1, 1))

    return np.hstack(betas).astype(np.float32)


def calculate_betas_for_x(X, covars):
    betas = run_regressions(X, covars)

    betas_shape_expected = (covars.shape[1], X.shape[1])

    assert betas.shape == betas_shape_expected, f'betas have shape {betas.shape} but should be {betas_shape_expected}'
    print(f'n_covariates x n_features matrix of shape {betas.shape} created')

    return betas


def calculate_betas_for_y(y, covars, method='logistic'):
    if method == 'linear':
        betas = LinearRegression().fit(covars, y).coef_.reshape(-1, 1)
    elif method == 'logistic':
        betas = LogisticRegression().fit(covars, y).coef_.reshape(-1, 1)
    else:
        raise ValueError(f'Method {method} for deconfounding y not recognised.')

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


def calculate_residuals_for_y(y, cov, betas):
    cov_dot_beta = np.dot(cov, betas)  # pulled into separate line for sanity checking dimensions

    assert cov_dot_beta.shape == y.shape, \
        f'Dot product of covariates and betas needs shape {y.shape} but is shape {cov_dot_beta.shape}'

    residuals = np.subtract(y, cov_dot_beta)

    return residuals.astype(np.float32)


def force_compute(hdf5_file, **kwargs):
    with h5py.File(hdf5_file) as f:
        X, y, rows, columns = read_ml(hdf5_file, f, **kwargs)
        X = X.compute()
        y = y.compute()

    return X, y, rows, columns


def parse_bool(bool_arg):
    bool_arg = bool_arg.lower()
    if bool_arg == 'false':
        bool_arg = False
    elif bool_arg == 'true':
        bool_arg = True
    else:
        raise ValueError(f'Arg {bool_arg} not recognised. Must be in ["True", "False"].')

    return bool_arg


def main(hdf5_file, covar_file, out, standardise=True, scaler=None, row_chunks=100,
         x_betas=None, y_betas=None, write_unadjusted=True):
    X, y, rows, columns = force_compute(hdf5_file)
    covars = parse_covars(covar_file, rows.IID.to_numpy())

    if standardise:
        # standardise with previous mean/sd from train split if given else use estimates from current data
        if scaler is not None:
            covars = scaler.transform(covars)
        else:
            scaler = StandardScaler()
            covars = scaler.fit_transform(covars)

    if x_betas is None:
        print('\nCalculating betas for X')
        x_betas = calculate_betas_for_x(X, covars)

    if y_betas is None:
        print('\nCalculating betas for y')
        y_betas = calculate_betas_for_y(y.squeeze(), covars, method='linear')

    print('\nAdjusting X/y using betas')
    X_residuals = calculate_residuals_for_x(X, covars, x_betas)
    y_residuals = calculate_residuals_for_y(y, covars, y_betas)

    # write rows/columns
    print('\n--> Saving to hdf5')
    rows.to_hdf(out, 'rows')
    columns.to_hdf(out, 'cols')

    if write_unadjusted:
        # write original x/y
        da.to_hdf5(out, {'x': da.from_array(X.astype(np.float16),
                                            chunks=(row_chunks, X.shape[1]))},
                chunks=(row_chunks, X.shape[1]))
        da.to_hdf5(out, {'y': da.from_array(y.astype(np.float16), chunks=(row_chunks, 1))},
                chunks=(row_chunks, 1))

    # write adjusted x/y
    da.to_hdf5(out, {'x_adjusted': da.from_array(X_residuals.astype(np.float16),
                                                 chunks=(row_chunks, X_residuals.shape[1]))},
               chunks=(row_chunks, X_residuals.shape[1]))
    da.to_hdf5(out, {'y_adjusted': da.from_array(y_residuals.astype(np.float16), chunks=(row_chunks, 1))},
               chunks=(row_chunks, 1))

    return scaler, x_betas, y_betas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read X/y/rows/columns from .hdf5 and adjust for covariates in covar file',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--train', type=str, default='',
                        help='Train hdf5 file with X/y/rows/cols')
    parser.add_argument('--test', type=str, default=None,
                        help='Test hdf5 file with X/y/rows/cols')
    parser.add_argument('--covar', type=str, default='',
                        help='Covar file, tab delimited with IID column')
    parser.add_argument('--out_train', type=str,
                        help='Full path for train hdf5 file including extension')
    parser.add_argument('--out_test', type=str,
                        help='Full path for test hdf5 file including extension')
    parser.add_argument('--standardise_covars', type=str, default='True',
                        help='z-transform covariates before regression')
    parser.add_argument('--write_unadjusted', type=str, default='True',
                        help='If True, also write the original X/y to out_train and out_test')
    args = parser.parse_args()

    standardise, write_unadjusted = [parse_bool(x) for x in [args.standardise_covars, args.write_unadjusted]]
    print('\n--> Beginning Train...')
    scaler, x_betas, y_betas = main(
        hdf5_file=args.train,
        covar_file=args.covar,
        out=args.out_train,
        standardise=standardise,
        write_unadjusted=write_unadjusted
    )

    if args.test is not None:
        print('\n--> Beginning Test...')
        _ = main(
            hdf5_file=args.test,
            covar_file=args.covar,
            out=args.out_test,
            standardise=standardise,
            scaler=scaler,
            x_betas=x_betas,
            y_betas=y_betas,
            write_unadjusted=write_unadjusted
        )
