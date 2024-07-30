import pytest
import numpy as np
from dask.distributed import Client
from daxos.boruta import boruta, get_importance, nanrankdata, assign_hits, do_tests, fdrcorrection, add_shadows_get_imps

@pytest.fixture
def dask_client():
    client = Client(n_workers=2, threads_per_worker=2)
    yield client
    client.close()

def test_boruta(dask_client):
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    columns = [f'col_{i}' for i in range(X.shape[1])]
    params = {'objective': 'binary:logistic', 'max_depth': 3,
              'eta': 0.01, 'n_boost_round': 10, 'subsample': 0.8, 'colsample_bytree': 0.8}

    support, support_weak = boruta(X, y, columns, params, dask_client)
    assert len(support) == X.shape[1]
    assert len(support_weak) == X.shape[1]

def test_get_importance(dask_client):
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    columns = [f'col_{i}' for i in range(X.shape[1])]
    params = {'objective': 'binary:logistic', 'max_depth': 3,
              'eta': 0.01, 'n_boost_round': 10, 'subsample': 0.8, 'colsample_bytree': 0.8}
    importance = get_importance(X, y, columns, params, dask_client)
    assert len(importance) == X.shape[1]

def test_nanrankdata():
    X = np.array([[np.nan, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    ranks = nanrankdata(X, axis=1)
    assert ranks.shape == X.shape

def test_assign_hits():
    hit_reg = np.zeros(5, dtype=int)
    cur_imp = (np.array([0.1, 0.4, 0.2, 0.3, np.nan]), np.array([0.1, 0.4, 0.2, 0.3, np.nan]))
    imp_sha_max = 0.25
    hit_reg = assign_hits(hit_reg, cur_imp, imp_sha_max)
    assert hit_reg.tolist() == [0, 1, 0, 1, 0]

def test_do_tests():
    dec_reg = np.array([0, 0, 0, 0, 0])
    hit_reg = np.array([1, 2, 1, 3, 1])
    _iter = 5
    two_step = True
    alpha = 0.05
    dec_reg = do_tests(dec_reg, hit_reg, _iter, two_step, alpha)
    assert dec_reg.shape == (5,)

def test_fdrcorrection():
    pvals = np.array([0.01, 0.04, 0.03, 0.002, 0.005])
    reject, pvals_corrected = fdrcorrection(pvals)
    assert len(reject) == len(pvals)
    assert len(pvals_corrected) == len(pvals)

def test_add_shadows_get_imps(dask_client):
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    dec_reg = np.zeros(10, dtype=int)
    columns = [f'col_{i}' for i in range(X.shape[1])]
    params = {'objective': 'binary:logistic', 'max_depth': 3,
              'eta': 0.01, 'n_boost_round': 10, 'subsample': 0.8, 'colsample_bytree': 0.8}

    imp_real, imp_sha = add_shadows_get_imps(X, y, dec_reg, columns, params, dask_client)
    assert imp_real.shape == (X.shape[1],)
    assert len(imp_sha) >= 5

if __name__ == "__main__":
    pytest.main()

