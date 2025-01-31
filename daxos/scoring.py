from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as smf
from datetime import datetime
import logging
import scipy.stats as sp
import pandas as pd
import numpy as np
import argparse
import time
import os


def fit_rf(df, covars, y_pred='y_pred', rf_trees=1000, iter=50, training_scorer='neg_root_mean_squared_error',
           nsubsample=5000, params=None):
    """

    Parameters
    ----------
    df
    covars
    y_pred
    rf_trees
    iter
    training_scorer
    nsubsample
    params

    Returns
    -------

    """
    rf = RandomForestRegressor(n_estimators=rf_trees)

    if params is None:
        hp_warning = """
        ################################################ WARNING #################################################
        Using default hyper-parameter search for RandomForestRegressor
        Default setting is a reasonably broad search for RFs, so recommend iter set as 30-100
        Sampled hyper-parameter distributions roughly correspond to:
          - max_depth = typically around 3 to 10, skewed binomial distribution, median 4
          - min_samples_split = typically around 2 to 8, skewed binomial distribution, median 3
          - max_features = uniformly sampled between 3 and 8
        ##########################################################################################################
        """
        logging.warning(hp_warning)

        params = dict(max_depth=sp.nbinom(n=2, p=0.45, loc=2),
                      min_samples_split=sp.nbinom(n=1.5, p=0.5, loc=2),
                      max_features=sp.randint(3, 9))

    rfcv = RandomizedSearchCV(estimator=rf, param_distributions=params, cv=5, scoring=training_scorer,
                              refit='neg_root_mean_squared_error', return_train_score=True, n_iter=iter,
                              n_jobs=-1, verbose=1)

    df_train = df.sample(n=nsubsample, replace=False).loc[:, [y_pred] + covars]
    X_train = df_train.drop(columns=y_pred).values
    y_train = df_train[y_pred].values

    logging.info(f'Running hyperparameter search on {nsubsample} rows')
    rfcv.fit(X_train, y_train)

    logging.info(f'Best CV score of {rfcv.best_score_:.2f} achieve with params {rfcv.best_params_}')
    logging.info(f'Refitting with best params from CV')
    X, y = df.loc[:, covars].values, df[y_pred].values
    rf = RandomForestRegressor(n_estimators=rf_trees, **rfcv.best_params_, n_jobs=-1)
    rf.fit(X, y)
    fv = rf.predict(X)
    res = y - fv

    return fv, res


def fit_ols(df, covars, y_pred='y_pred'):
    """

    Parameters
    ----------
    df
    covars
    y_pred

    Returns
    -------

    """
    str_covars = ' + '.join(covars)
    mod = smf.ols(f'{y_pred} ~ {str_covars}', data=df).fit(disp=0)

    return mod.fittedvalues.values, mod.resid.values


def auc(data, y_true='y_true', y_pred='y_pred', covars=None, model='ols', **rf_kwargs):
    """

    Parameters
    ----------
    data: pandas DataFrame

    y_true: str, default 'y_true'

    y_pred: str, default 'y_pred'

    covars: list of str or None, default None

    model: str, default 'ols'

    Returns
    -------
    AUC: float
        AUC(y_true, y_pred), after any adjusting of y_pred for covars
    R2: float
        R2 for y_pred ~ covars, where model for mapping X to y is given above
    FittedValues: np.array like
        Predictions on X
    Residuals: np.array like
        y_pred - fitted values

    """
    if covars is None:
        a = roc_auc_score(data[y_true].values, data[y_pred].values)
        r, fv, res = None, None, None
    else:
        if model == 'ols':
            fv, res = fit_ols(data, covars, y_pred=y_pred)
        elif model == 'rf':
            fv, res = fit_rf(data, covars, y_pred=y_pred, **rf_kwargs)
        else:
            raise ValueError('"model" must be in ["ols", "rf"]')

        a = roc_auc_score(data[y_true].values, res)
        r = r2_score(data[y_true].values, fv)

    return a, r, fv, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adjust predicted probabilities for covariates',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--predictions', '-p', type=str, default='',
                        help='File with predictions. Required columns: ["IID", "y_true", "y_pred"]. '
                             'Alternative names for IID, y_true and y_pred columns must be passed separately. '
                             'Assumed .csv file. Delimiter must be passed separately otherwise.')
    parser.add_argument('--covars', '-c', type=str, default='',
                        help='Tab-delimited file with covariates. Required columns: ["IID"]. '
                             'Covar file should be pre-processed for ML (no NAs, dummy-encoded categoricals etc.)')
    parser.add_argument('--model', '-m', type=str, default='ols',
                        help='Which model to use to regress-off covariates. Must be in ["ols", "rf", "both"], '
                             'where ols is linear regression and rf is random forest regression.')
    parser.add_argument('--ypred', type=str, default='y_pred',
                        help='Column name for predictions in file passed to "--predictions"')
    parser.add_argument('--ytrue', type=str, default='y_true',
                        help='Column name for outcome in file passed to "--predictions"')
    parser.add_argument('--id', type=str, default='IID',
                        help='Column name for IDs in file passed to "--predictions"')
    parser.add_argument('--sep', type=str, default=',',
                        help='Separator for columns passed to "--predictions". Default is "," (csv).')
    parser.add_argument('--out', type=str, default=None,
                        help='File name to save adjusted predictions')
    args = parser.parse_args()

    t0 = time.time()
    if args.out is not None:
        base, ext = os.path.splitext(args.out)
        logfile = f'{base}.log'
    else:
        logfile = f'scoring_{datetime.now().strftime("%d_%m_%Y_timestamp_%H_%M_%S")}.log'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        filename=logfile, filemode='a')

    # read and merge predictions and covariates
    df = pd.read_csv(args.predictions, sep=args.sep)
    cov = pd.read_csv(args.covars, sep='\t')

    if args.id != "IID":
        df = df.rename(columns={args.id: "IID"})

    cov = cov.drop(columns=['FID'], errors='ignore')

    df = df.merge(cov, on='IID', how='inner')
    cov_list = cov.set_index('IID').columns.tolist()

    # run adjusted AUCs
    logging.info('\n--> Checking baseline AUC before adjusting for covariates')
    baseline_auc, *_ = auc(df, y_true=args.ytrue, y_pred=args.ypred, covars=None)
    logging.info(f'AUC before adjusting for covariates: {baseline_auc:.4f}')

    models = [args.model] if args.model != "both" else ["ols", "rf"]

    adjusted_values = []
    for m in models:
        logging.info(f'\n--> Checking AUC after adjusting for covariates using an {m.upper()} regression model')
        adjusted_auc, r2, fv, res = auc(df, y_true=args.ytrue, y_pred=args.ypred, covars=cov_list, model=m)
        logging.info(f'AUC after adjusting for covariates: {adjusted_auc:.4f}')
        logging.info(f"Variance in predictions explained by covariates under {m.upper()} model (sklearn's R2 score): {r2:.4f}")
        adjusted_values.append(res)

    if args.out is not None:
        adjusted_values = (pd.concat([df.loc[:, ['IID', args.ytrue, args.ypred]],
                                      pd.DataFrame(np.hstack([a.reshape(-1, 1) for a in adjusted_values]),
                                                   columns=models)], axis=1)
                           .rename(columns=dict(zip(models + [args.ytrue, args.ypred],
                                                    [f'{s.upper()}_adjusted_y_pred' for s in models] +
                                                    ['y_true', 'y_pred'])))
                           .to_csv(args.out, index=False))

    t1 = time.time()
    t2 = t1 - t0
    logging.info(f'\nTime taken to process on CPU: '
                 f'{t2 // 3600 % 24:.2f} hours, {t2 // 60 % 60:.2f} minutes, {t2 % 60:.2f} seconds')
