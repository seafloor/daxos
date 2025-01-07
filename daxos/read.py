from .utils import set_numpy_dtype
import dask.array as da
import h5py
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import os


def load_booster(model_path):
    bst = xgb.Booster({})
    bst.load_model(model_path)
    if isinstance(bst, xgb.Booster):
        print(f'--> Loaded booster with {len(bst.trees_to_dataframe().Tree.unique())} trees')
        if bst.feature_names is not None:
            print(f'Booster trained on {len(bst.feature_names)} predictors')
        else:
            print('Booster trained on unknown number of predictors (feature_names not saved)')
        if hasattr(bst, 'get_score'):
            print(f'{len(bst.get_score().keys())} are included in the model')
        else:
            print('No features included in the model')

    return bst


def save_booster(bst, out_dir):
    bst.save_model(out_dir)
    if os.path.exists(out_dir):
        print(f'Saved XGBoost model in {out_dir}')


def raw_to_hdf5(f, out_name, nrows, row_chunks=100, check_output=False, read_raw_chunk_size=1000, dtype='float16'):
    X = read_raw_chunked(f, nrows, read_chunk_size=read_raw_chunk_size, dask_row_chunk_size=row_chunks, dtype=dtype)

    fam, y = read_info_from_raw(f, row_chunks)
    colnames = read_plink_colnames(f)

    write_dask_ml(X, y, row_chunks, out_name)
    write_dask_ml_info_from_raw(fam, colnames, out_name)

    if check_output:
        with h5py.File(out_name, mode='r') as in_file:
            _ = read_dask_ml_file(in_file, X.shape[1], row_chunks=row_chunks)


def read_raw_chunked(f, nrows, read_chunk_size=1000, dask_row_chunk_size=100, dtype='float16'):
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


def read_info_from_raw(f, row_chunks, max_rows=None, verbose=True):
    dtypes = {'FID': str, 'IID': str, 'PAT': str, 'MAT': str, 'SEX': np.float16, 'PHENOTYPE': np.float16}
    fam = pd.read_csv(f, dtype=dtypes, usecols=np.arange(0, 6), delimiter='\s+', nrows=max_rows)
    y = da.from_array(fam.iloc[:, 5].to_numpy(np.float16).reshape(-1, 1), chunks=(row_chunks, 1))

    if y.max() == 2:
        if verbose:
            print('\n--> Max value for y is 2 - dropping 1/2 coding to 0/1 coding by subtracting 1 from all rows')
        y = y - 1

    return fam, y


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


def print_summary(X, y):
    print('\nFirst 10 rows and predictors in X:')
    print(X[:10, :10].compute())
    print(X)

    print('\nFirst 10 rows in y:')
    print(y[:10].compute())
    print(y)


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


def write_dask_meta(rows, columns, path, row_key='rows', col_key='cols'):
    rows.to_hdf(path, row_key)
    columns.to_hdf(path, col_key)


def write_ml(X, y, rows, columns, path, row_chunks=100, **kwargs):
    write_dask_meta(rows, columns, path)
    write_dask_ml(X, y, row_chunks, path, **kwargs)


def subset_hdf5s(in_path, out_path, x_key='x', y_key='y', row_ids=None, col_ids=None):
    if all([row_ids is None, col_ids is None]):
        raise ValueError('Neither row or column IDs were passed')

    with h5py.File(in_path, 'r') as f:
        X, y, rows, columns = read_ml(in_path, f, x_key=x_key, y_key=y_key)

        if row_ids is not None:
            X, y, rows = subset_hdf5_rows(X, y, rows, row_ids)
        if col_ids is not None:
            X, columns = subset_hdf5_columns(X, columns, col_ids)

        row_chunks = min(10, X.shape[0]) if X.shape[0] < 100 else 100
        
        write_ml(X, y, rows, columns, out_path, row_chunks=row_chunks, x_key=x_key, y_key=y_key)


def subset_hdf5_rows(X, y, rows, row_ids):
    row_bool = rows.IID.isin(row_ids).to_numpy()
    out_rows = rows.loc[row_bool, :].copy()
    X = X[row_bool, :]
    y = y[row_bool, :]

    return X, y, out_rows


def subset_hdf5_columns(X, columns, col_ids):
    col_bool = columns.squeeze().str.split('_', expand=True).iloc[:, 0].isin(col_ids).to_numpy()
    out_columns = columns.loc[col_bool, :]
    X = X[:, col_bool]

    return X, out_columns
