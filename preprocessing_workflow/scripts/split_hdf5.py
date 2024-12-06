import argparse
import pandas as pd
import dask.array as da
import h5py
import sys
import os


def print_summary(X, y):
    print('\nFirst 10 rows and predictors in X:')
    print(X[:10, :10].compute())
    print(X)

    print('\nFirst 10 rows in y:')
    print(y[:10].compute())
    print(y)


def read_ml(container_name, file_object, verbose=True, **kwargs):
    rows, columns = read_dask_ml_info_file(container_name, verbose=verbose)
    X, y = read_dask_ml_file(f=file_object, col_chunks=columns.shape[0], verbose=verbose, **kwargs)

    return X, y, rows, columns


def read_dask_ml_file(f, col_chunks=-1, row_chunks=100, verbose=True, x_key='x', y_key='y'):
    if verbose:
        print('\n--> Attempting to read dask ml hdf5 file as dask arrays...')
        print('Specifying row chunks as {} and col_chunks for X as {}'.format(row_chunks, col_chunks))

    X = da.from_array(f[x_key], chunks=(row_chunks, col_chunks))
    y = da.from_array(f[y_key], chunks=(row_chunks, 1))

    if verbose:
        print_summary(X, y)

    return X, y


def read_dask_ml_info_file(f, verbose=True, row_key='rows', column_key='cols'):
    if verbose:
        print('\n--> Attempting to read dask ml hdf5 info file as pandas DataFrames...')

    rows = pd.read_hdf(f, row_key)
    cols = pd.read_hdf(f, column_key)

    if verbose:
        print('\nSample of row information:')
        print(rows.head())
        print('\nSample of column information:')
        print(cols.head())

    return rows, cols


def subset_hdf5_rows(X, y, rows, row_ids):
    print('\n--> Subsetting hdf5 file...')
    row_bool = rows.IID.isin(row_ids).to_numpy()
    out_rows = rows.loc[row_bool, :].copy()
    X = X[row_bool, :]
    y = y[row_bool, :]

    return X, y, out_rows


def write_dask_ml(X, y, row_chunks, out_name, x_col_chunks=None, x_key='x', y_key='y'):
    print('\n--> Saving {} row chunks to {} as hdf5'.format(row_chunks, out_name))
    print('Writing X and y data')

    x_col_chunks = X.shape[1] if x_col_chunks is None else x_col_chunks
    da.to_hdf5(out_name, {x_key: X}, chunks=(row_chunks, x_col_chunks))
    da.to_hdf5(out_name, {y_key: y}, chunks=(row_chunks, 1))


def write_dask_meta(rows, columns, path, row_key='rows', col_key='cols'):
    rows.to_hdf(path, row_key)
    columns.to_hdf(path, col_key)


def write_ml(X, y, rows, columns, path, row_chunks=100, **kwargs):
    write_dask_meta(rows, columns, path)
    write_dask_ml(X, y, row_chunks, path, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subset hdf5 files by rows',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--in_path', type=str,
                        help='Full path for existing hdf5 file')
    parser.add_argument('--out_path', type=str,
                        help='Full path for hdf5 file to write to')
    parser.add_argument('--ids', default=None, type=str,
                        help='Path to IIDs (one row per ID)')
    parser.add_argument('--row_chunks', default=100, type=int,
                        help='Row chunk size for output hdf5 file')
    parser.add_argument('--xkey', default='x', type=str,
                        help='The key for genotype data in the hdf5 file')
    parser.add_argument('--ykey', default='y', type=str,
                        help='The key for the outcome in the hdf5 file')
    args = parser.parse_args()

    ids = pd.read_csv(args.ids, header=None, dtype=str).squeeze("columns").to_numpy() if args.ids is not None else None
    assert ids.shape[0] > 1, f'ID file too short: shape {ids.shape}'
    print(f'--> Read {ids.shape[0]} IDs for subsetting. Sample: {ids[:5].flatten()}')

    with h5py.File(args.in_path) as f:
        X, y, rows, columns = read_ml(args.in_path, f, x_key=args.xkey, y_key=args.ykey)
        X, y, rows = subset_hdf5_rows(X, y, rows, ids)

        write_ml(X, y, rows, columns, args.out_path, row_chunks=args.row_chunks, x_key=args.xkey, y_key=args.ykey)

        print('\n--> Reading split file back in as a final check...')
        _ = read_dask_ml_info_file(args.out_path, verbose=True, row_key='rows', column_key='cols')

        with h5py.File(args.out_path, mode='r') as in_file:
            _ = read_dask_ml_file(in_file, X.shape[1], row_chunks=args.row_chunks)
