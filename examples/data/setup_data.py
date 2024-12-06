import numpy as np
import dask.array as da

def create_snp_names(n=10):
    snp_columns = zip(np.random.randint(10000, 1000000, size=n),
                      np.random.choice(['A', 'C', 'G', 'T'], replace=True, size=n))

    return [f'rs{i}_{s}' for i, s in snp_columns]

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

def create_simple_sim(n=1000, p=100, seed=123):
    np.random.seed(seed)

    n_cases = int(n / 2)
    n_controls = n - n_cases
    odds_ratios = np.exp(np.random.normal(loc=0, scale=0.1, size=p))
    case_maf = np.random.uniform(low=0.05, high=0.5, size=p)
    control_maf = (case_maf / (odds_ratios * (1 - case_maf) + case_maf))
    y = (np.hstack((np.ones(n_cases), np.zeros(n_controls)))
           .astype(int)
           .reshape(n, 1))

    snp_names = create_snp_names(p)
    X = np.vstack(
            (np.array([np.random.binomial(2, maf, n_cases) for maf in case_maf]).T,
             np.array([np.random.binomial(2, maf, n_controls) for maf in control_maf]).T))
    X = np.hstack((y, X))

    return X, y, snp_names

def save_hdf5(out_name, X, y, snp_names):
    # create and save X and y
    X = da.from_array(X, chunks=(100, 100))
    y = da.from_array(y, chunks=(100, 1))
    da.to_hdf5(out_name, {'x': X}, chunks=(100, 100))
    da.to_hdf5(out_name, {'y': y}, chunks=(100, 1))

    # create and save rows and cols
    rows = create_dummy_fam_file(1000)
    rows.to_hdf(out_name, 'rows')
    
    cols = create_snp_names(100)
    pd.Series(cols).to_frame().to_hdf(out_name, 'cols')

if __name__ == '__main__':
    X, y, snp_names = create_simple_sim(n=1000, p=100)
    out_file = 'examples/data/example_data.hdf5'
    save_hdf5(out_file, X, y, snp_names)
    if os.path.exists(out_file):
        print(f'Data saved to {out_file}')
