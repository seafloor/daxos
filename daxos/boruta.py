"""
Adapted from borutapy: Daniel Homola

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
"""
import numpy as np
import scipy.stats as sp
import dask.array as da
import sys
import os
import xgboost as xgb
from .crossvalidate import persist_daskdmatrix, fit_dask_xgb

def boruta(X, y, columns, params, client, importance='shap', shap_sumstat='mean', max_iter=10, perc=100, alpha=0.05,
           two_step=True, row_chunks=100, train_split=0.7, train_or_test='test', **fit_kwargs):
    """
    :returns
    dataframe with snp and label "tentative/
    """
    n_sample, n_feat = X.shape
    _iter = 1

    # Early stopping vars
    dec_reg = np.zeros(n_feat, dtype=int)
    hit_reg = np.zeros(n_feat, dtype=int)
    imp_history = np.zeros(n_feat, dtype=float)
    sha_max_history = []

    while np.any(dec_reg == 0) and _iter < max_iter:
        print(f'Running boruta iteration {_iter}')
        cur_imp = add_shadows_get_imps(X, y, dec_reg, columns, params, client, importance=importance,
                                       shap_sumstat=shap_sumstat, row_chunks=row_chunks, train_split=train_split,
                                       train_or_test=train_or_test, **fit_kwargs)
        imp_sha_max = np.percentile(cur_imp[1], perc)
        sha_max_history.append(imp_sha_max)
        imp_history = np.vstack((imp_history, cur_imp[0]))
        hit_reg = assign_hits(hit_reg, cur_imp, imp_sha_max)
        dec_reg = do_tests(dec_reg, hit_reg, _iter, two_step, alpha)

        if _iter < max_iter:
            _iter += 1

    confirmed = np.where(dec_reg == 1)[0]
    tentative = np.where(dec_reg == 0)[0]
    tentative_median = np.median(imp_history[1:, tentative], axis=0)
    tentative_confirmed = np.where(tentative_median > np.median(sha_max_history))[0]
    tentative = tentative[tentative_confirmed]

    n_features_ = confirmed.shape[0]
    support_ = np.zeros(n_feat, dtype=bool)
    support_[confirmed] = 1
    support_weak_ = np.zeros(n_feat, dtype=bool)
    support_weak_[tentative] = 1

    ranking_ = np.ones(n_feat, dtype=int)
    ranking_[tentative] = 2
    selected = np.hstack((confirmed, tentative))
    not_selected = np.setdiff1d(np.arange(n_feat), selected)
    imp_history_rejected = imp_history[1:, not_selected] * -1

    if not_selected.shape[0] > 0:
        iter_ranks = nanrankdata(imp_history_rejected, axis=1)
        rank_medians = np.nanmedian(iter_ranks, axis=0)
        ranks = nanrankdata(rank_medians, axis=0)

        if tentative.shape[0] > 0:
            ranks = ranks - np.min(ranks) + 3
        else:
            ranks = ranks - np.min(ranks) + 2
        ranking_[not_selected] = ranks
    else:
        support_ = np.ones(n_feat, dtype=bool)

    importance_history_ = imp_history

    return support_, support_weak_

def push_and_train(client, X, y, params, colnames, **fit_kwargs):
    dtrain = persist_daskdmatrix(client, X, y, feature_names=None, manually_map_to_workers=False, method='persist')
    bst, history = fit_dask_xgb(client, dtrain, params, **fit_kwargs)
    return bst, dtrain

def get_importance(X, y, columns, params, client, importance='shap', shap_sumstat='mean', row_chunks=100,
                   train_split=0.7, train_or_test='test', **fit_kwargs):
    if train_split < 1:
        train_idx = np.random.choice(np.arange(X.shape[0]), size=int(train_split * X.shape[0]), replace=False)
        test_idx = np.delete(np.arange(X.shape[0]), train_idx)
        X_train = da.from_array(X[train_idx, :], chunks=(row_chunks, X.shape[1]))
        y_train = da.from_array(y[train_idx].reshape(-1, 1), chunks=(row_chunks, 1))
        X_test = da.from_array(X[test_idx, :], chunks=(row_chunks, X.shape[1]))
        y_test = da.from_array(y[test_idx].reshape(-1, 1), chunks=(row_chunks, 1))
    else:
        X_train = da.from_array(X, chunks=(row_chunks, X.shape[1]))
        y_train = da.from_array(y.reshape(-1, 1), chunks=(row_chunks, 1))

    bst, dtrain = push_and_train(client, X_train, y_train, params, colnames=columns, **fit_kwargs)

    if importance == 'shap':
        if train_or_test == 'train':
            shap = xgb.dask.predict(client, bst, dtrain, pred_contribs=True)
        elif train_or_test == 'test':
            dtest = persist_daskdmatrix(client, X_test, y_test, feature_names=None, manually_map_to_workers=False,
                                        method='persist')
            shap = xgb.dask.predict(client, bst, dtest, pred_contribs=True)
        else:
            raise ValueError('Must be train or test')

        if shap_sumstat == 'mean':
            # note last value is the bias term so dropped here
            imp = np.abs(shap).mean(axis=0)[:-1]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    del dtrain
    if 'dtest' in globals():
        del dtest

    return imp

def nanrankdata(X, axis=1):
    ranks = sp.mstats.rankdata(X, axis=axis)
    ranks[np.isnan(X)] = np.nan
    return ranks

def assign_hits(hit_reg, cur_imp, imp_sha_max):
    cur_imp_no_nan = cur_imp[0]
    cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
    hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
    hit_reg[hits] += 1
    return hit_reg

def do_tests(dec_reg, hit_reg, _iter, two_step, alpha):
    active_features = np.where(dec_reg >= 0)[0]
    hits = hit_reg[active_features]

    to_accept_ps = sp.binom.sf(hits - 1, _iter, .5).flatten()
    to_reject_ps = sp.binom.cdf(hits, _iter, .5).flatten()

    if two_step:
        to_accept = fdrcorrection(to_accept_ps, alpha=alpha)[0]
        to_reject = fdrcorrection(to_reject_ps, alpha=alpha)[0]
        to_accept2 = to_accept_ps <= alpha / float(_iter)
        to_reject2 = to_reject_ps <= alpha / float(_iter)
        to_accept *= to_accept2
        to_reject *= to_reject2
    else:
        to_accept = to_accept_ps <= alpha / float(len(dec_reg))
        to_reject = to_reject_ps <= alpha / float(len(dec_reg))

    to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
    to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

    dec_reg[active_features[to_accept]] = 1
    dec_reg[active_features[to_reject]] = -1
    return dec_reg

def fdrcorrection(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, pvals_sortind)
    nobs = len(pvals_sorted)
    ecdffactor = np.arange(1, nobs + 1) / float(nobs)

    reject = pvals_sorted <= ecdffactor * alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1] = 1

    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[pvals_sortind] = pvals_corrected
    reject_ = np.empty_like(reject)
    reject_[pvals_sortind] = reject
    return reject_, pvals_corrected_

def add_shadows_get_imps(X, y, dec_reg, columns, params, client, importance='shap', shap_sumstat='mean',
                         row_chunks=100, train_split=0.7, train_or_test='test', **fit_kwargs):
    x_cur_ind = np.where(dec_reg >= 0)[0]
    x_cur = np.copy(X[:, x_cur_ind])
    x_cur_w = x_cur.shape[1]
    x_sha = np.copy(x_cur)

    while (x_sha.shape[1] < 5):
        x_sha = np.hstack((x_sha, x_sha))

    np.apply_along_axis(np.random.shuffle, 0, x_sha)

    imp = get_importance(np.hstack((x_cur, x_sha)), y, columns, params, client, importance=importance,
                         shap_sumstat=shap_sumstat, row_chunks=row_chunks, train_split=train_split,
                         train_or_test=train_or_test, **fit_kwargs)

    imp_sha = imp[x_cur_w:]
    imp_real = np.zeros(X.shape[1])
    imp_real[:] = np.nan
    imp_real[x_cur_ind] = imp[:x_cur_w]

    return imp_real, imp_sha
