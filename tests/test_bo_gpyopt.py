import warnings

from GPy.kern import *

from paramopt.optimizers.gpyopt import BayesianOptimizer
from utils import f1, f2, search_1d, search_2d

warnings.filterwarnings('ignore')


def test_bo_gpy_1d_lcb():
    bo = BayesianOptimizer(
        savedir='tests/output/bo_gpy_1d_lcb',
        kernel=RBF(input_dim=1, variance=1, lengthscale=0.5)*Bias(input_dim=1, variance=1),
        acqfunc='LCB',
        acquisition_weight=2,
        random_seed=71)
    search_1d(bo, f1)


def test_bo_gpy_1d_ei():
    bo = BayesianOptimizer(
        savedir='tests/output/bo_gpy_1d_ei',
        kernel=RBF(input_dim=1, variance=1, lengthscale=0.5)*Bias(input_dim=1, variance=1),
        acqfunc='EI',
        acquisition_jitter=0.0,
        acquisition_maximize=True,
        random_seed=71)
    search_1d(bo, f1)


def test_bo_gpy_2d_lcb():
    bo = BayesianOptimizer(
        savedir='tests/output/bo_gpy_2d_lcb',
        kernel=RBF(input_dim=2, variance=1, lengthscale=5.0)*Bias(input_dim=2, variance=1),
        acqfunc='LCB',
        acquisition_weight=2,
        random_seed=71)
    search_2d(bo, f2)


def test_bo_gpy_2d_ei():
    bo = BayesianOptimizer(
        savedir='tests/output/bo_gpy_2d_ei',
        kernel=RBF(input_dim=2, variance=1, lengthscale=5.0)*Bias(input_dim=2, variance=1),
        acqfunc='EI',
        acquisition_jitter=0.0,
        acquisition_maximize=True,
        random_seed=71)
    search_2d(bo, f2)
