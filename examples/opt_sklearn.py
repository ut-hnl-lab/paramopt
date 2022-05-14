from sklearn.gaussian_process.kernels import *

from paramopt.optimizers.sklearn import BayesianOptimizer
from paramopt.acquisitions import UCB, EI
from utils import f1, f2, search_1d, search_2d


def slbo1d_ucb():
    bo = BayesianOptimizer(
        savedir='test_slbo1d-ucb',
        kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
        acqfunc=UCB(c=2.0),
        random_seed=71)
    search_1d(bo, f1)


def slbo1d_ei():
    bo = BayesianOptimizer(
        savedir='test_slbo1d-ei',
        kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
        acqfunc=EI(xi=0.0),
        random_seed=71)
    search_1d(bo, f1)


def slbo2d_ucb():
    bo = BayesianOptimizer(
        savedir='test_slbo2d-ucb',
        kernel=RBF(length_scale=5.0) * ConstantKernel() + WhiteKernel(),
        acqfunc=UCB(c=2.0),
        random_seed=71)
    search_2d(bo, f2)


def slbo2d_ei():
    bo = BayesianOptimizer(
        savedir='test_slbo2d-ei',
        kernel=RBF(length_scale=5.0) * ConstantKernel() + WhiteKernel(),
        acqfunc=EI(xi=0.0),
        random_seed=71)
    search_2d(bo, f2)


if __name__ == '__main__':
    slbo1d_ucb()
    slbo1d_ei()
    slbo2d_ucb()
    slbo2d_ei()
