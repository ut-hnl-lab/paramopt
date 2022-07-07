<h1 align="center"> Param Opt </h1>
<h3 align="center">Param Opt helps the researcher quickly and easily explore optimal experimental parameters.</h3>

## Table of Contents
* [Overview](#overview)
* [Preparation](#preparation)
* [Quick Start](#quick-start)
    * [Defining Target Parameters](#defining-target-parameters)
    * [Creating Dataset](#creating-dataset)
    * [Preparing GPR Model and Acquisition Function](#preparing-gpr-model-and-acquisition-function)
    * [Optimizing Parameters](#optimizing-parameters)
* [Other Useful Features](#other-useful-features)
    * [GPR with Hyperparameter Auto-adjustment Ability](#gpr-with-hyperparameter-auto-adjustment-ability)
    * [GIF Creation from Plot pngs](#gif-creation-from-plot-pngs)
* [License](#license)

## Overview
Bayesian optimization is used for adjusting process parameters (= experimental parameters), such as instrument settings, chemical formulation rates, hyperparameters for machine learning models, and more.

**Param Opt** is a useful python package that is responsible for not only bayesian model training and prediction, but also reading and writing data and visualizing the optimization process.

## Preparation
Install Param Opt via pip:
```
pip install git+https://github.com/ut-hnl-lab/paramopt.git
```
The following packages are also required:
* Matplotlib
* Natsort
* Numpy
* Pandas
* Pillow
* Scipy
* Scikit-learn

## Quick Start
Here is an example of optimizing a combination of two parameters.

### Defining Target Parameters
Define parameters to be adjusted.
```python
param1 = ProcessParameter(name="Heating Temperature", values=[200, 240, 280])
param2 = ProcessParameter(name="Heating Time", values=[30, 60, 120, 240])
```
`name` is the parameter name and `values` is a list of possible values of the parameter.

Then, define a exploration space consisting of the parameters.
```python
space = ExplorationSpace([param1, param2])
```

### Creating Dataset
Create a dataset consisting of an explanatory variables with `X_names` and objective variables with `Y_names`.
```python
dataset = Dataset(X_names=space.names, Y_names="Evaluation")
```
Basically, X_names is passed the parameter namew, and Y_names is passed the name of the evaluations.

The dataset is managed by the `BayesianOptimizer` class described below.

### Preparing GPR Model and Acquisition Function
Use `sklearn.gaussian_process.GaussianProcessRegressor` for the GPR model.
Acquisition functions are provided in this package.
```python
model = GaussianProcessRegressor(
        kernel=RBF()*ConstantKernel()+WhiteKernel(),
        normalize_y=True)
acquisition = UCB(1.0)
```

### Optimizing Parameters
Let's optimize parameters in the bayesian optimization loop.
Here is a function that simulates an experiment and returns an evaluation value for a given parameter combination.
```python
def experiment(x1, x2):
    return x1*np.sin(x1) + x2*np.cos(x2)
```

The optimization flow is as follows:
```python
# Define optimizer
bo = BayesianOptimizer(
    workdir=here, exploration_space=space, dataset=dataset, model=model,
    acquisition=acquisition, objective_fn=experiment, random_seed=71)

# For max iterations:
for i in range(10):
    # Get better combination of parameters
    next_x = bo.suggest()
    # Get evaluation score with experiments
    y = experiment(*next_x)
    # Update optimizer
    bo.update(next_x, y, label=f"#{i+1}")
    # Check process with some plots
    bo.plot()
```

## Other Useful Features

### GPR with Hyperparameter Auto-adjustment Ability
Sometimes GPR does not predict well like this:

In this case, let's replace it with a model that automatically adjusts the hyperparameters.

```python
def gpr_generator(exp, nro):
    return GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=(10**-exp, 10**exp)) \
                * ConstantKernel() \
                + WhiteKernel(),
        normalize_y=True, n_restarts_optimizer=nro)

model = AutoHPGPR(
    workdir=here, exploration_space=space, gpr_generator=gpr_generator,
    exp=list(range(1, 6)), nro=list(range(0, 10)))
```

The result is

### GIF Creation from Plot pngs
Create a GIF movie from the obtained plot images

```python
paths = select_images()  # Opens a GUI dialog
create_gif(paths)
```

# License
MIT license.
