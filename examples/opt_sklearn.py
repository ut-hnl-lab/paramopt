from sklearn.gaussian_process.kernels import *

from paramopt import GPR, UCB, EI
from .function import *
from .optimizer import *


def gpr1d_ucb():
    gpr = GPR(
        savedir='test_gpr1d-ucb',
        kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
        acqfunc=UCB(c=2.0),
        random_seed=71)
    search_1d(gpr, f1)


def gpr1d_ei():
    gpr = GPR(
        savedir='test_gpr1d-ei',
        kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
        acqfunc=EI(xi=0.0),
        random_seed=71)
    search_1d(gpr, f1)


def gpr2d_ucb():
    gpr = GPR(
        savedir='test_gpr2d-ucb',
        kernel=RBF(length_scale=5.0) * ConstantKernel() + WhiteKernel(),
        acqfunc=UCB(c=2.0),
        random_seed=71)
    search_2d(gpr, f2)


def gpr2d_ei():
    gpr = GPR(
        savedir='test_gpr2d-ei',
        kernel=RBF(length_scale=5.0) * ConstantKernel() + WhiteKernel(),
        acqfunc=EI(xi=0.0),
        random_seed=71)
    search_2d(gpr, f2)


if __name__ == '__main__':
    gpr1d_ucb()
    gpr1d_ei()
    gpr2d_ucb()
    gpr2d_ei()
