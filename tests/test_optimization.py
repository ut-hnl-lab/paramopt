from pathlib import Path
import numpy as np
import pytest
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


def test_base():
    workdir = Path.cwd()/'tests'/'output'/'explore_base'
    exspace = ExplorationSpace([ProcessParameter("Parameter 1", list(range(11)))])
    dataset = Dataset(exspace.names, "Evaluation")
    acquisition = UCB(1.0)

    bo = BaseOptimizer(
        workdir=workdir,
        exploration_space=exspace,
        dataset=dataset,
        acquisition=acquisition,
        objective_fn=f1,
    )

    with pytest.raises(NotImplementedError):
        explore_1d(bo)


def test_1d():
    workdir = Path.cwd()/'tests'/'output'/'explore_1d'
    exspace = ExplorationSpace([ProcessParameter("Parameter 1", list(range(11)))])
    dataset = Dataset(exspace.names, "Evaluation")
    model = GaussianProcessRegressor(
            kernel=RBF(length_scale=1, length_scale_bounds='fixed') \
                  * ConstantKernel() \
                  + WhiteKernel(),
            normalize_y=True)
    acquisition = UCB(1.0)

    bo = BayesianOptimizer(
        workdir=workdir,
        exploration_space=exspace,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f1,
        random_seed=71
    )

    repr(bo)
    explore_1d(bo)


def test_2d():
    workdir = Path.cwd()/'tests'/'output'/'explore_2d'
    exspace = ExplorationSpace([
        ProcessParameter("Parameter 1", list(range(150, 255, 5))),
        ProcessParameter("Parameter 2", list(range(10, 220, 10))),
    ])
    dataset = Dataset(exspace.names, "Evaluation")
    model = GaussianProcessRegressor(
            kernel=RBF(length_scale=50, length_scale_bounds='fixed') \
                  * ConstantKernel() \
                  + WhiteKernel(),
            normalize_y=True)
    acquisition = UCB(1.0)

    bo = BayesianOptimizer(
        workdir=workdir,
        exploration_space=exspace,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f2,
        random_seed=71
    )

    repr(bo)
    explore_2d(bo)


def test_1d_autohp():
    workdir = Path.cwd()/'tests'/'output'/'explore_1d_autohp'
    exspace = ExplorationSpace([ProcessParameter("Parameter 1", list(range(11)))])

    def gpr_generator(ls, nro):
        return GaussianProcessRegressor(
            kernel=RBF(length_scale=ls, length_scale_bounds='fixed') \
                    * ConstantKernel() \
                    + WhiteKernel(),
            normalize_y=True,
            n_restarts_optimizer=nro
        )

    model = AutoHPGPR(
        workdir=workdir,
        exploration_space=exspace,
        gpr_generator=gpr_generator,
        ls=[0.1, 1, 10],
        nro=[0, 1, 2]
    )

    dataset = Dataset(exspace.names, "Evaluation")
    acquisition = UCB(2.0)

    bo = BayesianOptimizer(
        workdir=workdir,
        exploration_space=exspace,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f1,
        random_seed=71
    )

    repr(bo)
    explore_1d(bo)


def test_2d_autohp():
    workdir = Path.cwd()/'tests'/'output'/'explore_2d_autohp'
    exspace = ExplorationSpace([
        ProcessParameter("Parameter 1", list(range(150, 255, 5))),
        ProcessParameter("Parameter 2", list(range(10, 220, 10))),
    ])

    def gpr_generator(ls, nro):
        return GaussianProcessRegressor(
            kernel=RBF(length_scale=ls, length_scale_bounds='fixed') \
                    * ConstantKernel() \
                    + WhiteKernel(),
            normalize_y=True,
            n_restarts_optimizer=nro
        )

    model = AutoHPGPR(
        workdir=workdir,
        exploration_space=exspace,
        gpr_generator=gpr_generator,
        stop_fitting_score=0.98,
        ls=[0.1, 1, 10],
        nro=[0, 1, 2]
    )

    dataset = Dataset(exspace.names, "Evaluation")
    acquisition = UCB(2.0)

    bo = BayesianOptimizer(
        workdir=workdir,
        exploration_space=exspace,
        dataset=dataset,
        model=model,
        acquisition=acquisition,
        objective_fn=f2,
        random_seed=71
    )

    repr(bo)
    explore_2d(bo)
