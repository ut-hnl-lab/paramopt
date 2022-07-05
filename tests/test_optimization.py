from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from paramopt.acquisitions.ucb import UCB
from paramopt.extensions.hpadjust import AutoHPGPR
from paramopt.optimizers import BaseOptimizer
from paramopt.optimizers.sklearn import BayesianOptimizer
from paramopt.structures.dataset import Dataset
from paramopt.structures.parameter import ExplorationSpace, ProcessParameter


def f1(x):
    return x * np.sin(x)


def f2(x1, x2):
    return (x1 * np.sin(x1*0.05+np.pi) + x2 * np.cos(x2*0.05)) * 0.1


def explore_1d(bo: BaseOptimizer, noisy: bool = True) -> None:
    bo.update(0, f1(0), label=str(0))
    bo.update(1, f1(1), label=str(0))
    bo.update(10, f1(10), label=str(0))
    bo.plot()

    for i in range(10):
        next_x, = bo.suggest()
        y = f1(next_x)
        if noisy:
            y += np.random.normal(scale=0.1)
        bo.update(next_x, y, label=str(i+1))
        bo.plot()


def explore_2d(bo: BaseOptimizer, noisy: bool = True) -> None:
    bo.update((150, 10), f2(150, 10), label=str(0))
    bo.update((150, 210), f2(150, 210), label=str(0))
    bo.update((250, 10), f2(150, 10), label=str(0))
    bo.update((250, 210), f2(250, 210), label=str(0))
    bo.update((200, 110), f2(200, 110), label=str(0))
    bo.plot()

    for i in range(10):
        next_x = bo.suggest()
        y = f2(*next_x)
        if noisy:
            y += np.random.normal(scale=1.0)
        bo.update(next_x, y, label=str(i+1))
        bo.plot()


def test_1d():
    workdir = Path.cwd()/'tests'/'output'/'explore_1d'
    space = ExplorationSpace([ProcessParameter("Parameter 1", list(range(11)))])
    dataset = Dataset(space.names, "Evaluation")
    model = GaussianProcessRegressor(
            kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
            normalize_y=True)
    acquisition = UCB(1.0)

    bayesian_optimizer = BayesianOptimizer(
        workdir=workdir,
        exploration_space=space,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f1,
        random_seed=71
    )
    explore_1d(bayesian_optimizer)


def test_2d():
    workdir = Path.cwd()/'tests'/'output'/'explore_2d'
    space = ExplorationSpace([
        ProcessParameter("Parameter 1", list(range(150, 255, 5))),
        ProcessParameter("Parameter 2", list(range(10, 220, 10))),
    ])
    dataset = Dataset(space.names, "Evaluation")
    model = GaussianProcessRegressor(
            kernel=RBF() * ConstantKernel() + WhiteKernel(),
            normalize_y=True)
    acquisition = UCB(1.0)

    bayesian_optimizer = BayesianOptimizer(
        workdir=workdir,
        exploration_space=space,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f2,
        random_seed=71
    )
    explore_2d(bayesian_optimizer)


def test_1d_autohp():
    workdir = Path.cwd()/'tests'/'output'/'explore_1d_autohp'
    space = ExplorationSpace([ProcessParameter("Parameter 1", list(range(11)))])

    def gpr_generator(exp, nro):
        return GaussianProcessRegressor(
            kernel=RBF(length_scale_bounds=(10**-exp, 10**exp)) \
                    * ConstantKernel() \
                    + WhiteKernel(),
            normalize_y=True,
            n_restarts_optimizer=nro
        )

    model = AutoHPGPR(
        workdir=workdir,
        exploration_space=space,
        gpr_generator=gpr_generator,
        exp=list(range(1, 6)),
        nro=list(range(0, 10))
    )

    dataset = Dataset(space.names, "Evaluation")
    acquisition = UCB(1.0)

    bayesian_optimizer = BayesianOptimizer(
        workdir=workdir,
        exploration_space=space,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f1,
        random_seed=71
    )
    explore_1d(bayesian_optimizer)


def test_2d_autohp():
    workdir = Path.cwd()/'tests'/'output'/'explore_2d_autohp'
    space = ExplorationSpace([
        ProcessParameter("Parameter 1", list(range(150, 255, 5))),
        ProcessParameter("Parameter 2", list(range(10, 220, 10))),
    ])

    def gpr_generator(exp, nro):
        return GaussianProcessRegressor(
            kernel=RBF(length_scale_bounds=(10**-exp, 10**exp)) \
                    * ConstantKernel() \
                    + WhiteKernel(),
            normalize_y=True,
            n_restarts_optimizer=nro
        )

    model = AutoHPGPR(
        workdir=workdir,
        exploration_space=space,
        gpr_generator=gpr_generator,
        exp=list(range(1, 6)),
        nro=list(range(0, 10))
    )

    dataset = Dataset(space.names, "Evaluation")
    acquisition = UCB(1.0)

    bayesian_optimizer = BayesianOptimizer(
        workdir=workdir,
        exploration_space=space,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f2,
        random_seed=71
    )
    explore_2d(bayesian_optimizer)
