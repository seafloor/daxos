import argparse
import numpy as np
import h5py
import dask.array as da
import sys
import os
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from dask_xgboost_slurm import read, deconfound
from dask_xgboost_slurm.utils import parse_bool
from sklearn.preprocessing import StandardScaler


def force_lazy_loading_compute(hdf5_file, **kwargs):
    with h5py.File(hdf5_file) as f:
        X, y, rows, columns = read.read_ml(hdf5_file, f, **kwargs)
        X = X.compute()
        y = y.compute()

    return X, y, rows, columns


def main(hdf5_file, covar_file, out, standardise=True, scaler=None, row_chunks=100, x_betas=None, y_betas=None):
    X, y, rows, columns = force_lazy_loading_compute(hdf5_file)
    covars = deconfound.parse_covars(covar_file, rows.IID.to_numpy())

    if standardise:
        if scaler is None:
            scaler = StandardScaler()
            covars = scaler.fit_transform(covars)
        else:
            covars = scaler.transform(covars)

    if x_betas is None:
        print('\nCalculating betas for X')
        x_betas = deconfound.calculate_betas_for_x(X, covars)

    if y_betas is None:
        print('\nCalculating betas for y')
        y_betas = deconfound.calculate_betas_for_y(y.squeeze(), covars, method='linear')

    print('\nAdjusting X/y using betas')
    X_residuals = deconfound.calculate_residuals_for_x(X, covars, x_betas)
    y_residuals = deconfound.calculate_residuals_for_y(y, covars, y_betas)

    # write rows/columns
    print('\nSaving to hdf5')
    rows.to_hdf(out, 'rows')
    columns.to_hdf(out, 'cols')

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
    parser = argparse.ArgumentParser(description='Read X/y/rows/columns from .hdf5 and adjust for covars in covar file',
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
    args = parser.parse_args()

    standardise = parse_bool(args.standardise_covars)
    print('\n--> Beginning Train...')
    scaler, x_betas, y_betas = main(args.train, args.covar, args.out_train, standardise)

    if args.test is not None:
        X, y, rows, columns = force_lazy_loading_compute(args.test)
        covars = deconfound.parse_covars(args.covar, rows.IID.to_numpy())

        print('\n--> Beginning Test...')
        scaler, x_betas, y_betas = main(args.test, args.covar, args.out_test, standardise,
                                        scaler=scaler, x_betas=x_betas, y_betas=y_betas)
