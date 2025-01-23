from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from split_hdf5 import read_ml
from split_ids import read_covars, check_covars
import argparse
import numpy as np
import h5py
import dask.array as da
import sys
import os


def get_slurm_cores():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        # Use cpus-per-task if defined
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    elif "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        # Use job CPUs per node if defined
        return sum(map(int, os.environ["SLURM_JOB_CPUS_PER_NODE"].split(',')))
    else:
        # Default to os.cpu_count() if not running under SLURM
        return os.cpu_count()


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


def process_column_with_shm(args):
    column, X_shm_name, covars_shm_name, X_shape, covars_shape, X_dtype, covars_dtype = args

    # Attach to shared memory
    X_shm = shared_memory.SharedMemory(name=X_shm_name)
    covars_shm = shared_memory.SharedMemory(name=covars_shm_name)

    X = np.ndarray(X_shape, dtype=X_dtype, buffer=X_shm.buf)
    covars = np.ndarray(covars_shape, dtype=covars_dtype, buffer=covars_shm.buf)

    # Process column as usual
    lr = LinearRegression()
    not_nan_bool = ~np.isnan(X[:, column])
    lr.fit(covars[not_nan_bool, :], X[not_nan_bool, column])
    return lr.coef_.reshape(-1, 1)


def run_regressions(X, covars, cores=5, report_interval=10000):
    # Create shared memory for X and covars
    X_shared = shared_memory.SharedMemory(create=True, size=X.nbytes)
    covars_shared = shared_memory.SharedMemory(create=True, size=covars.nbytes)

    # Copy data to shared memory
    X_shm = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
    X_shm[:] = X[:]
    covars_shm = np.ndarray(covars.shape, dtype=covars.dtype, buffer=covars_shared.buf)
    covars_shm[:] = covars[:]

    # Arguments only include indices
    args = [(column, X_shared.name, covars_shared.name, X.shape, covars.shape, X.dtype, covars.dtype) 
            for column in range(X.shape[1])]

    print('\n--> Beginning processing log for covariate adjustment...')
    with ProcessPoolExecutor(max_workers=cores) as executor:
        betas = list(executor.map(process_column_with_shm, args))

    # Free shared memory
    X_shared.close()
    X_shared.unlink()
    covars_shared.close()
    covars_shared.unlink()

    return np.hstack(betas).astype(np.float32)


def calculate_betas_for_x(X, covars, cores=5):
    betas = run_regressions(X, covars, cores=cores)

    betas_shape_expected = (covars.shape[1], X.shape[1])

    assert betas.shape == betas_shape_expected, f'betas have shape {betas.shape} but should be {betas_shape_expected}'
    print(f'n_covariates x n_features matrix of shape {betas.shape} created')

    return betas


def calculate_betas_for_y(y, covars, method='logistic'):
    # Shared memory setup (if not already done)
    covars_shared = shared_memory.SharedMemory(create=True, size=covars.nbytes)
    covars_shm = np.ndarray(covars.shape, dtype=covars.dtype, buffer=covars_shared.buf)
    covars_shm[:] = covars[:]

    y_shared = shared_memory.SharedMemory(create=True, size=y.nbytes)
    y_shm = np.ndarray(y.shape, dtype=y.dtype, buffer=y_shared.buf)
    y_shm[:] = y[:]

    # Perform regression (method-specific logic as in your code)
    if method == 'linear':
        betas = LinearRegression().fit(covars_shm, y_shm).coef_.reshape(-1, 1)
    elif method == 'logistic':
        betas = LogisticRegression().fit(covars_shm, y_shm).coef_.reshape(-1, 1)
    else:
        raise ValueError(f'Method {method} for deconfounding y not recognised.')

    # Cleanup shared memory
    covars_shared.close()
    covars_shared.unlink()
    y_shared.close()
    y_shared.unlink()

    return betas


def calculate_residuals_for_x(X, covars, betas):
    # Setup shared memory for X
    X_shared = shared_memory.SharedMemory(create=True, size=X.nbytes)
    X_shm = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
    X_shm[:] = X[:]

    # Setup shared memory for covars
    covars_shared = shared_memory.SharedMemory(create=True, size=covars.nbytes)
    covars_shm = np.ndarray(covars.shape, dtype=covars.dtype, buffer=covars_shared.buf)
    covars_shm[:] = covars[:]

    # Setup shared memory for betas
    betas_shared = shared_memory.SharedMemory(create=True, size=betas.nbytes)
    betas_shm = np.ndarray(betas.shape, dtype=betas.dtype, buffer=betas_shared.buf)
    betas_shm[:] = betas[:]

    # Compute dot product
    cov_dot_beta = np.dot(covars_shm, betas_shm)

    # Check dimensions for safety
    assert cov_dot_beta.shape == X_shm.shape, \
        f"Dot product shape mismatch: Expected {X_shm.shape}, got {cov_dot_beta.shape}"

    # Calculate residuals
    residuals = X_shm - cov_dot_beta

    # Cleanup shared memory
    X_shared.close()
    X_shared.unlink()
    covars_shared.close()
    covars_shared.unlink()
    betas_shared.close()
    betas_shared.unlink()

    return residuals.astype(np.float32)


def calculate_residuals_for_y(y, covars, betas):
    # Shared memory setup
    covars_shared = shared_memory.SharedMemory(create=True, size=covars.nbytes)
    covars_shm = np.ndarray(covars.shape, dtype=covars.dtype, buffer=covars_shared.buf)
    covars_shm[:] = covars[:]

    y_shared = shared_memory.SharedMemory(create=True, size=y.nbytes)
    y_shm = np.ndarray(y.shape, dtype=y.dtype, buffer=y_shared.buf)
    y_shm[:] = y[:]

    betas_shared = shared_memory.SharedMemory(create=True, size=betas.nbytes)
    betas_shm = np.ndarray(betas.shape, dtype=betas.dtype, buffer=betas_shared.buf)
    betas_shm[:] = betas[:]

    # Calculate residuals
    cov_dot_beta = np.dot(covars_shm, betas_shm)
    residuals = y_shm - cov_dot_beta

    # Cleanup shared memory
    covars_shared.close()
    covars_shared.unlink()
    y_shared.close()
    y_shared.unlink()
    betas_shared.close()
    betas_shared.unlink()

    return residuals.astype(np.float32)


def force_compute(hdf5_file, **kwargs):
    with h5py.File(hdf5_file, 'r') as f:
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
         x_betas=None, y_betas=None, write_unadjusted=True, cores=5):
    
    if cores is None:
        cores = get_slurm_cores()

    print(f"\n--> Using {cores} cores for processing.")
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
        x_betas = calculate_betas_for_x(X, covars, cores=cores)

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
    parser.add_argument('--cores', type=int, default=None,
                        help='N cores available on machine. For regression parallelism.')
    args = parser.parse_args()

    standardise, write_unadjusted = [parse_bool(x) for x in [args.standardise_covars, args.write_unadjusted]]
    print('\n--> Beginning Train...')
    scaler, x_betas, y_betas = main(
        hdf5_file=args.train,
        covar_file=args.covar,
        out=args.out_train,
        standardise=standardise,
        write_unadjusted=write_unadjusted,
        cores=args.cores
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
            write_unadjusted=write_unadjusted,
            cores=args.cores
        )
