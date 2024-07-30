import xgboost as xgb
import numpy as np
import dask.array as da
import pandas as pd
import h5py

# Create a simple XGBoost model and save it
def create_dummy_xgb_model():
    X = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)
    features = create_snp_names()
    dtrain = xgb.DMatrix(X, label=y, feature_names=features)
    params = {'objective': 'binary:logistic', 'max_depth': 2, 'eta': 1}
    bst = xgb.train(params, dtrain, num_boost_round=2)
    bst.save_model('./dummy_model.xgb')

def create_dummy_fam_file(rows=100):
    ids = np.random.randint(10000, 20000, size=rows)
    fam = pd.DataFrame({
        'FID': ids,
        'IID': ids, 
        'PAT': np.zeros(rows).astype(int),
        'MAT': np.zeros(rows).astype(int),
        'SEX': np.random.choice([1, 2], size=rows),
        'PHENOTYPE': np.random.choice([1, 2], size=rows)
    })

    return fam

def create_snp_names(n=10):
    snp_columns = zip(np.random.randint(10000, 1000000, size=n),
                      np.random.choice(['A', 'C', 'G', 'T'], replace=True, size=n))

    return [f'rs{i}_{s}' for i, s in snp_columns]

def create_dummy_snp_data(rows=100, columns=10):
    # setup rsids names
    snp_columns = create_snp_names(columns)

    # create snp data
    snp_data = pd.DataFrame(np.random.randint(0, 3, size=(rows, columns)), columns=snp_columns)
    
    return snp_data

# Create a dummy PLINK file
def create_dummy_plink_file():
     # create core ids
    core_data = create_dummy_fam_file()

    # create snp data
    snp_data = create_dummy_snp_data()

    # merge and save
    df = pd.concat([core_data, snp_data], axis=1)
    df.to_csv('./dummy_plink.raw', sep=' ', index=False)

# Create a dummy HDF5 file
def create_dummy_hdf5_file():
    out_name = './dummy_data.hdf5'

    # create and save X and y
    X = da.random.random((100, 10), chunks=(10, 10))
    y = da.random.choice([0, 1], size=(100, 1), replace=True, chunks=(10, 1))
    da.to_hdf5(out_name, {'x': X}, chunks=(10, 10))
    da.to_hdf5(out_name, {'y': y}, chunks=(10, 1))

    # create and save rows and cols
    rows = create_dummy_fam_file()
    rows.to_hdf(out_name, 'rows')
    
    cols = create_snp_names(10)
    pd.Series(cols).to_frame().to_hdf(out_name, 'cols')

if __name__ == "__main__":
    create_dummy_xgb_model()
    create_dummy_plink_file()
    create_dummy_hdf5_file()
