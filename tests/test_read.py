import pytest
import dask.array as da
import pandas as pd
import numpy as np
import h5py
import xgboost as xgb
import os
from daxos.read import (load_booster, save_booster, raw_to_hdf5, read_raw_chunked, read_x_from_raw, 
                        read_info_from_raw, read_ml, read_dask_ml_file, read_dask_ml_info_file, 
                        read_plink_colnames, write_dask_ml_info_from_raw,  write_dask_ml, 
                        write_dask_meta, write_ml, subset_hdf5s, subset_hdf5_rows,  subset_hdf5_columns)

# note all test assume dummy test data is in the tests/data subdir
def test_load_booster():
    model_path = 'tests/data/dummy_model.xgb'
    booster = load_booster(model_path)
    assert isinstance(booster, xgb.Booster)

def test_save_booster():
    model_path = 'tests/data/dummy_model.xgb'
    out_path = 'test_model.xgb'
    booster = load_booster(model_path)
    save_booster(booster, out_path)
    assert os.path.exists(out_path)

def test_raw_to_hdf5():
    f = 'tests/data/dummy_plink.raw'
    out_name = 'tests/data/output.hdf5'
    nrows = 100
    raw_to_hdf5(f, out_name, nrows, check_output=True)
    assert os.path.exists(out_name)

def test_read_raw_chunked():
    f = 'tests/data/dummy_plink.raw'
    nrows = 100
    X = read_raw_chunked(f, nrows)
    assert isinstance(X, da.Array)
    assert X.shape == (100, 10)

def test_read_x_from_raw():
    f = 'tests/data/dummy_plink.raw'
    X = read_x_from_raw(f)
    assert isinstance(X, da.Array)
    assert X.shape == (100, 10)

def test_read_info_from_raw():
    f = 'tests/data/dummy_plink.raw'
    row_chunks = 100
    fam, y = read_info_from_raw(f, row_chunks)
    assert isinstance(fam, pd.DataFrame)
    assert isinstance(y, da.Array)

def test_read_ml():
    with h5py.File('tests/data/dummy_data.hdf5', 'r') as f:
        X, y, rows, cols = read_ml('tests/data/dummy_data.hdf5', f)
        assert isinstance(X, da.Array)
        assert isinstance(y, da.Array)
        assert isinstance(rows, pd.DataFrame)
        assert isinstance(cols, pd.DataFrame)

def test_read_dask_ml_file():
    with h5py.File('tests/data/dummy_data.hdf5', 'r') as f:
        X, y = read_dask_ml_file(f)
        assert isinstance(X, da.Array)
        assert isinstance(y, da.Array)

def test_read_dask_ml_info_file():
    f = 'tests/data/dummy_data.hdf5'
    rows, cols = read_dask_ml_info_file(f)
    assert isinstance(rows, pd.DataFrame)
    assert isinstance(cols, pd.DataFrame)

def test_read_plink_colnames():
    f = 'tests/data/dummy_plink.raw'
    colnames = read_plink_colnames(f)
    assert isinstance(colnames, list)
    assert len(colnames) == 16

def test_write_dask_ml_info_from_raw():
    df = pd.DataFrame({'FID': [1, 2], 'IID': [1, 2], 'PAT': [0, 0], 'MAT': [0, 0],
                       'SEX': [1, 2], 'PHENOTYPE': [1, 2]})
    colnames = [f'rs{i}' for i in range(1, 11)]
    out_name = 'tests/data/output_info.hdf5'
    write_dask_ml_info_from_raw(df, colnames, out_name)
    assert os.path.exists(out_name)

def test_write_dask_ml():
    X = da.from_array(np.random.rand(100, 10), chunks=(50, 10))
    y = da.from_array(np.random.rand(100, 1), chunks=(50, 1))
    out_name = 'tests/data/output_ml.hdf5'
    write_dask_ml(X, y, 50, out_name)
    assert os.path.exists(out_name)

def test_write_dask_meta():
    rows = pd.DataFrame(np.arange(100))
    cols = pd.DataFrame(np.arange(10))
    path = 'tests/data/output_meta.hdf5'
    write_dask_meta(rows, cols, path)
    assert os.path.exists(path)

def test_write_ml():
    X = da.from_array(np.random.rand(100, 10), chunks=(50, 10))
    y = da.from_array(np.random.rand(100, 1), chunks=(50, 1))
    rows = pd.DataFrame(np.arange(100))
    cols = pd.DataFrame(np.arange(10))
    path = 'tests/data/output_ml_all.hdf5'
    write_ml(X, y, rows, cols, path, 50)
    assert os.path.exists(path)

def test_subset_hdf5s():
    in_path = 'tests/data/dummy_data.hdf5'
    out_path = 'tests/data/subset.hdf5'
    row_ids = np.array([12688, 13492, 18122, 17249, 19277, 17565, 19960, 14231, 15764])
    col_ids = np.array(['rs905664', 'rs113441', 'rs33092', 'rs342088'])
    subset_hdf5s(in_path, out_path, row_ids=row_ids, col_ids=col_ids)
    assert os.path.exists(out_path)

def test_subset_hdf5_rows():
    X = da.from_array(np.random.rand(100, 10), chunks=(50, 10))
    y = da.from_array(np.random.rand(100, 1), chunks=(50, 1))
    rows = pd.DataFrame(np.arange(100), columns=['IID'])
    row_ids = np.arange(50)
    X_sub, y_sub, rows_sub = subset_hdf5_rows(X, y, rows, row_ids)
    assert X_sub.shape[0] == 50
    assert y_sub.shape[0] == 50
    assert len(rows_sub) == 50

def test_subset_hdf5_columns():
    X = da.from_array(np.random.rand(100, 10), chunks=(50, 10))
    columns = pd.DataFrame([f'col{i}' for i in range(10)])
    col_ids = [f'col{i}' for i in range(5)]
    X_sub, cols_sub = subset_hdf5_columns(X, columns, col_ids)
    assert X_sub.shape[1] == 5
    assert len(cols_sub) == 5

def teardown_module(module):
    # List of files to remove
    files_to_remove = [
        'tests/data/saved_model.xgb',
        'tests/data/output.hdf5',
        'tests/data/output_info.hdf5',
        'tests/data/output_ml.hdf5',
        'tests/data/output_meta.hdf5',
        'tests/data/output_ml_all.hdf5',
        'tests/data/subset.hdf5'
    ]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    pytest.main()
