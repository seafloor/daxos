from itertools import compress
import dask.array as da
import pandas as pd
import numpy as np
import time
import h5py
import argparse
import os

def set_numpy_dtype(dtype_str):
    dtype_options = [np.float16, np.float32, np.float64]
    dtype_numpy = list(compress(dtype_options, [dtype_str.lower() == s for s in ('float16', 'float32', 'float64')]))
    if len(dtype_numpy) != 1:
        raise ValueError(f'NumPy dtypes selected as {dtype_numpy}')

    return dtype_numpy[0]

def read_x_from_raw(f, max_rows=None, skip_rows=0, colnames=None, row_chunks=100, dtype='float16', verbose=True):
    start_time = time.time()
    if verbose:
        print('\n--> Reading from file: {}'.format(f))
        print('Reading column names and setting dtypes')
    colnames = read_plink_colnames(f) if colnames is None else colnames
    dtype_numpy = set_numpy_dtype(dtype)

    if verbose:
        print('Reading in raw file as numpy array')
    X = np.genfromtxt(f, usecols=np.arange(6, len(colnames)), dtype=dtype_numpy, skip_header=1 + skip_rows,
                      max_rows=max_rows)
    print(f'Loaded numpy array of size {X.nbytes / 1e9:.2g}GB in {(time.time() - start_time) / 60:.2f} minutes')

    if verbose:
        print('Converting to row-chunked dask array')
    X = da.from_array(X, chunks=(row_chunks, X.shape[1]))

    return X

def read_raw_chunked(f, nrows, read_chunk_size=1000, dask_row_chunk_size=100, dtype='float16'):
    """

    """
    print('\n################################################################################')
    print(f'\n--> Reading raw file {f} with {nrows} rows as chunked dask arrays')
    n_chunks = int(np.ceil(nrows / read_chunk_size))
    print(f'--> {n_chunks} row-chunks to be used of size {read_chunk_size}')
    print('\n################################################################################')

    X = []
    for i in np.arange(1, n_chunks + 1):
        chunk_start_idx = int((i - 1) * read_chunk_size)
        print(f'\n--> Reading chunk {int(i)} of {n_chunks} with {read_chunk_size} rows, starting from index {chunk_start_idx}')
        X.append(read_x_from_raw(f, max_rows=read_chunk_size, skip_rows=chunk_start_idx, row_chunks=dask_row_chunk_size,
                                 dtype=dtype, verbose=False))

    return da.concatenate(X, axis=0)

def read_info_from_raw(f, row_chunks, max_rows=None, verbose=True):
    dtypes = {'FID': str, 'IID': str, 'PAT': str, 'MAT': str, 'SEX': np.float16, 'PHENOTYPE': np.float16}
    fam = pd.read_csv(f, dtype=dtypes, usecols=np.arange(0, 6), delimiter='\s+', nrows=max_rows)
    y = da.from_array(fam.iloc[:, 5].to_numpy(np.float16).reshape(-1, 1), chunks=(row_chunks, 1))

    if y.max() == 2:
        if verbose:
            print('\n--> Max value for y is 2 - dropping 1/2 coding to 0/1 coding by subtracting 1 from all rows')
        y = y - 1

    return fam, y

def read_plink_colnames(f):
    colnames = pd.read_csv(f, nrows=1, header=None)

    return colnames[0].str.split('\s+').iat[0]

def write_dask_ml_info_from_raw(df, colnames, out_name):
    print('Writing row and column information')
    pd.Series(colnames[6:]).to_frame().to_hdf(out_name, 'cols')
    df.iloc[:, :6].reset_index(drop=True).to_hdf(out_name, 'rows')

def write_dask_ml(X, y, row_chunks, out_name, x_col_chunks=None, x_key='x', y_key='y'):
    print('\n--> Saving {} row chunks to {} as hdf5'.format(row_chunks, out_name))
    print('Writing X and y data')

    x_col_chunks = X.shape[1] if x_col_chunks is None else x_col_chunks
    da.to_hdf5(out_name, {x_key: X}, chunks=(row_chunks, x_col_chunks))
    da.to_hdf5(out_name, {y_key: y}, chunks=(row_chunks, 1))

def print_summary(X, y):
    print('\nFirst 10 rows and predictors in X:')
    print(X[:10, :10].compute())
    print(X)

    print('\nFirst 10 rows in y:')
    print(y[:10].compute())
    print(y)

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

def raw_to_hdf5(f, out_name, nrows, row_chunks=100, check_output=False, read_raw_chunk_size=1000, dtype='float16'):
    X = read_raw_chunked(f, nrows, read_chunk_size=read_raw_chunk_size, dask_row_chunk_size=row_chunks, dtype=dtype)

    fam, y = read_info_from_raw(f, row_chunks)
    colnames = read_plink_colnames(f)

    write_dask_ml(X, y, row_chunks, out_name)
    write_dask_ml_info_from_raw(fam, colnames, out_name)

    if check_output:
        _ = read_dask_ml_info_file(out_name, verbose=True, row_key='rows', column_key='cols')

        with h5py.File(out_name, mode='r') as in_file:
            _ = read_dask_ml_file(in_file, X.shape[1], row_chunks=row_chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert plink .raw files to dask .hdf5 files for machine learning',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--raw', type=str, default='',
                        help='Full path for input raw file')
    parser.add_argument('--nrows', type=int,
                        help='Number of individuals (nrow of fam file)')
    parser.add_argument('--dask_chunks', type=int, default=100,
                        help='Size of chunks in hdf5 file. Keeo low so multiples of it can be used for reading.')
    parser.add_argument('--read_raw_chunk_size', type=int, default=1000,
                        help='Number of rows to read from raw file in one go. Keep in the thousands to keep RAM low.')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='Numpy dtype to use for all columns. Must be in ["float16", "float32", "float64"].')

    args = parser.parse_args()

    if args.dtype not in ['float16', 'float32', 'float64']:
        raise ValueError('--dtype not recognised')

    print('\n--> Starting raw conversion to hdf5 with params: {}'.format(args))

    raw_file, raw_ext = os.path.splitext(args.raw)

    raw_to_hdf5(
        f = args.raw,
        out_name = f"{raw_file}.hdf5",
        nrows = args.nrows,
        row_chunks = args.dask_chunks,
        check_output = True,
        read_raw_chunk_size = args.read_raw_chunk_size,
        dtype = args.dtype
    )
