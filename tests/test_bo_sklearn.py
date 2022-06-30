from sklearn.gaussian_process.kernels import *

from paramopt.optimizers.sklearn import BayesianOptimizer
from paramopt.acquisitions import UCB, EI
from utils import f1, f2, search_1d, search_2d


def test_bo_sk_1d_ucb():
    bo = BayesianOptimizer(
        savedir='test/output/bo_sk_1d_ucb',
        kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
        acqfunc=UCB(c=2.0),
        random_seed=71)
    search_1d(bo, f1)


def test_bo_sk_1d_ei():
    bo = BayesianOptimizer(
        savedir='test/output/bo_sk_1d_ei',
        kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
        acqfunc=EI(xi=0.0),
        random_seed=71)
    search_1d(bo, f1)


def test_bo_sk_2d_ucb():
    bo = BayesianOptimizer(
        savedir='test/output/bo_sk_2d_ucb',
        kernel=RBF(length_scale=5.0) * ConstantKernel() + WhiteKernel(),
        acqfunc=UCB(c=2.0),
        random_seed=71)
    search_2d(bo, f2)


def test_bo_sk_2d_ei():
    bo = BayesianOptimizer(
        savedir='test/output/bo_sk_2d_ei',
        kernel=RBF(length_scale=5.0) * ConstantKernel() + WhiteKernel(),
        acqfunc=EI(xi=0.0),
        random_seed=71)
    search_2d(bo, f2)
