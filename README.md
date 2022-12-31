# ParamOpt
Python Library for Easy Bayesian Optimization.

## Install
```
pip install git+https://github.com/blue-no/paramopt.git
```

## Demo - 2D Exploration
```Python
from paramopt import UCB, BayesianOptimizer, ExplorationSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def obj_func(X):
    return -(X[0]-5)**2-(X[1]-50)**2/50

optimizer = BayesianOptimizer(
    regressor=GaussianProcessRegressor(kernel=C()*RBF(), normalize_y=True),
    exp_space=ExplorationSpace({
        'distance':
            {'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             'unit': 'mm'},
        'temperature':
            {'values': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
             'unit': '°C'}
    }),
    eval_name='quality',
    acq_func=UCB(c=1.0),
    obj_func=obj_func)

for i in range(10):
    next_params = optimizer.suggest()
    y = obj_func(next_params)
    optimizer.update(X=next_params, y=y, label=i)
    optimizer.plot_distribution()
    optimizer.plot_transition()
    optimizer.save_history(folder.joinpath('history.csv'))
```

**Author:** Kota AONO  
**License:** MIT License
