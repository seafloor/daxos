import numpy as np
import pandas as pd
import dask.array as da
import os
import argparse
import statsmodels.api as sm


def create_dummy_snp_names(p=10):
    snp_columns = zip(np.random.randint(10000, 1000000, size=p-2),
                      np.random.choice(['A', 'C', 'G', 'T'], replace=True, size=p-2))
    
    return [f'rs{i}_{s}' for i, s in snp_columns] + ['rs7412_T', 'rs429358_C']


def create_dummy_fam_file(n=100):
    ids = np.random.randint(10000, 20000, size=n)
    fam = pd.DataFrame({
        'FID': ids,
        'IID': ids, 
        'PAT': np.zeros(n).astype(int),
        'MAT': np.zeros(n).astype(int),
        'SEX': np.random.choice([1, 2], size=n),
        'PHENOTYPE': np.random.choice([1, 2], size=n)
    })

    return fam


def create_simple_sim(n=10000, p=100, seed=123):
    print(f'\n--> Creating simulation with {n} samples, {p} SNPs, and seed {seed}')
    np.random.seed(seed)

    n_cases = int(n / 2)
    n_controls = n - n_cases
    odds_ratios = np.hstack([np.repeat(1, p-2), np.array([3, 5])])
    case_maf = np.random.uniform(low=0.05, high=0.5, size=p)
    control_maf = (case_maf / (odds_ratios * (1 - case_maf) + case_maf))
    y = (np.hstack((np.ones(n_cases), np.zeros(n_controls)))
           .astype(int)
           .reshape(n, 1))
    
    print(f'Simulated odds ratios: {odds_ratios}')

    print('Generating SNP names...')
    snp_names = create_dummy_snp_names(p)

    print('Generating genotype matrix...')
    X = np.vstack(
            (np.array([np.random.binomial(2, maf, n_cases) for maf in case_maf]).T,
             np.array([np.random.binomial(2, maf, n_controls) for maf in control_maf]).T))

    print('Simulation data creation complete.')

    # double-check they are roughly correct
    check_ors_reasonable(X, y, odds_ratios)

    return X, y, snp_names


def save_hdf5(out_name, X, y, snp_names):
    print(f'\n--> Saving data to {out_name}')
    # Create and save X and y
    X = da.from_array(X, chunks=(100, 10))
    y = da.from_array(y, chunks=(100, 1))
    da.to_hdf5(out_name, {'x': X}, chunks=(100, 10))
    da.to_hdf5(out_name, {'y': y}, chunks=(100, 1))

    # Create and save rows and cols
    print('Creating dummy family file...')
    rows = create_dummy_fam_file(1000)
    rows.to_hdf(out_name, 'rows')
    
    print('Saving SNP names...')
    cols = pd.DataFrame({'SNP': snp_names})
    cols.to_hdf(out_name, 'cols')


def check_odds_ratios(X, y):
    # Ensure y is 1D
    y = y.ravel()

    # Initialize list to store odds ratios
    odds_ratios = []

    print(f"Performing univariable logistic regression for {X.shape[1]} predictors...")

    # Iterate through each predictor (column) in X
    for i in range(X.shape[1]):
        predictor = X[:, i]
        
        # Add a constant for the intercept term
        predictor_with_const = sm.add_constant(predictor)

        # Fit logistic regression model
        model = sm.Logit(y, predictor_with_const)
        result = model.fit(disp=False)  # Suppress fitting output

        # Compute and store the odds ratio
        odds_ratio = np.exp(result.params[1])  # Exponentiate the coefficient
        odds_ratios.append(odds_ratio)

        # Progress reporting for every 10th feature
        if i % 10 == 0:
            print(f"Processed predictor {i + 1}/{X.shape[1]}")

    print("Logistic regression complete.")

    return np.array(odds_ratios)


def check_ors_reasonable(X, y, expected_or, tol=0.5):
    print('\n--> Checking simulated ORs match expectations')
    actual_or = check_odds_ratios(X, y)

    in_range = np.abs(expected_or - actual_or) < tol

    if np.all(in_range):
        print(f'All ORs within {tol} of expected values')
    else:
        print(f'One or more ORs deviate by > {tol} from expected values:\n')
        print(actual_or)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create simulation data for genetic analysis.',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--n', type=int, default=10000, help='Number of samples (default: 1000)')
    parser.add_argument('--p', type=int, default=10, help='Number of SNPs (default: 100)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility (default: 123)')
    parser.add_argument('--output_file', type=str, default='example_data.hdf5', help='Output file name (default: example_data.hdf5)')
    args = parser.parse_args()

    print('\n--> Starting simulation data creation...')
    print(f'Parameters: n={args.n}, p={args.p}, seed={args.seed}, output_file={args.output_file}')

    X, y, snp_names = create_simple_sim(n=args.n, p=args.p, seed=args.seed)
    save_hdf5(args.output_file, X, y, snp_names)

    if os.path.exists(args.output_file):
        print(f'\nSimulation data saved successfully to {args.output_file}')
    else:
        print(f'\nError: Could not save data to {args.output_file}')
