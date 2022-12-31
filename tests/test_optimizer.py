import warnings
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import ConvergenceWarning

from paramopt import (UCB, AutoHyperparameterRegressor, BayesianOptimizer,
                      ExplorationSpace)

N_TRIAL = 10
INIT_PARAMS = [1, 10]
X1_VALS = list(range(0, 11, 1))
X2_VALS = list(range(0, 101, 10))

folder = Path('tests', 'result', 'optimizer')
folder.mkdir(exist_ok=True)


def obj_func(X):
    return -(X[0]-5)**2-(X[1]-50)**2/50


def test_optimization():
    optimizer = BayesianOptimizer(
        regressor=GaussianProcessRegressor(
            kernel=C()*RBF(10, 'fixed'),
            normalize_y=True
        ),
        exp_space=ExplorationSpace({
            'x1': {'values': X1_VALS, 'unit': 'u1'},
            'x2': {'values': X2_VALS, 'unit': 'u2'}
        }),
        eval_name='eval',
        acq_func=UCB(c=2.0),
        obj_func=obj_func,
        working_dir=folder,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(N_TRIAL):
            next_params = optimizer.suggest()
            y = obj_func(next_params)
            optimizer.update(next_params, y, label=f'normal_{i}')
            optimizer.plot_distribution()
            optimizer.plot_transition()
            optimizer.save_history(folder.joinpath('history-normal.csv'))


def test_autohp_optimization():
    regressor = AutoHyperparameterRegressor(
        hyperparams={
            'ls': list(range(10, 110, 10)),
            'nro': list(range(0, 11, 1))
        },
        regressor_factory=lambda autohp:
            GaussianProcessRegressor(
                kernel=C()*RBF(autohp.select('ls'), 'fixed'),
                normalize_y=True,
                n_restarts_optimizer=autohp.select('nro')
            )
    )

    optimizer = BayesianOptimizer(
        regressor=regressor,
        exp_space=ExplorationSpace({
            'x1': {'values': X1_VALS, 'unit': 'u1'},
            'x2': {'values': X2_VALS, 'unit': 'u2'}
        }),
        eval_name='eval',
        acq_func=UCB(c=2.0),
        obj_func=obj_func,
        working_dir=folder,
    )

    for i in range(N_TRIAL):
        next_params = optimizer.suggest()
        y = obj_func(next_params)
        optimizer.update(next_params, y, label=f'autohp_{i}')
        optimizer.plot_distribution()
        optimizer.plot_transition()
        optimizer.save_history(folder.joinpath('history-autohp.csv'))
        regressor.dump_hp_history(folder.joinpath('hp.csv'))
