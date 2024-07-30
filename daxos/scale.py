from .crossvalidate import fit_one_round_cv
from .read import print_summary
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np


def run_cv_and_platt_scale(client, X, y_adjusted, y_binary, params, n_fold=5, score_method='AUC',
                           manually_map_to_workers=False, **fit_kwargs):
    print('Re-running XGBoost CV on best params to get test fold predictions')
    scores, predictions, y_true = fit_one_round_cv(client, X, y_adjusted, params, n_fold, score_method,
                                                   manually_map_to_workers=manually_map_to_workers, **fit_kwargs)

    # concatenate predictions
    X_stacked = np.hstack(predictions).reshape(-1, 1)
    y_stacked = y_binary
    print('\nPredictions stacked to give new X for platt scaling:')
    print_summary(X_stacked, y_stacked)

    # fit model - make generic so doesn't have to be LR
    print('\nFitting model to XGBoost predictions')
    lr = LogisticRegression(penalty=None)
    lr.fit(X_stacked, y_stacked)

    y_pred = lr.predict_proba(X_stacked)[:, 1]
    print(f'Platt-scaled AUC on train set (biased - cannot be used to measure performance): '
          f'{roc_auc_score(y_stacked, y_pred):.2g}')

    return lr
