<div align="center">
<h1> Param Opt </h1>
<h4>Param Opt helps the researcher quickly and easily find best experimental parameters.</h4>

<img src="https://img.shields.io/badge/version-1.1.1-green"> <img src="https://img.shields.io/github/license/ut-hnl-lab/paramopt?color=yellow"> <img src="https://img.shields.io/badge/python-3.7%2B-blue">
</div>

## Table of Contents
* [Overview](#overview)
* [Installation](#installation)
* [Quick Start](#quick-start)
    * [Defining Target Parameters](#lidefining-target-parametersli)
    * [Creating Dataset](#licreating-datasetli)
    * [Preparing GPR Model and Acquisition Function](#lipreparing-gpr-model-and-acquisition-functionli)
    * [Optimizing Parameters](#lioptimizing-parametersli)
* [Other Useful Features](#other-useful-features)
    * [GPR with Hyperparameter Auto-adjustment Ability](#ligpr-with-hyperparameter-auto-adjustment-abilityli)
    * [GIF Creation from Plot pngs](#ligif-creation-from-plot-pngsli)
* [License](#license)

## Overview

<img src=https://user-images.githubusercontent.com/88641432/179461743-f48ec565-e66e-4089-976b-4d4aa9c3accb.png>

Bayesian optimization is used for adjusting process parameters (= experimental parameters), such as instrument settings, chemical formulation rates, hyperparameters for machine learning models, and more.

**Param Opt** is a useful python package that is responsible for not only bayesian model training and prediction, but also reading and writing data and visualizing the optimization process.

## Installation
Install Param Opt via pip:
```
pip install git+https://github.com/ut-hnl-lab/paramopt.git
```
The following packages are also required:
* Dacite
* Matplotlib
* Natsort
* Numpy
* Pandas
* Pillow
* Scipy
* Scikit-learn

## Quick Start
Here is an example of optimizing a combination of two parameters.

Firstly, import all necessary packages and define the current directory as a working directory.
```python
import pathlib
import numpy as np
from paramopt.structures import ProcessParameter, ExplorationSpace, Dataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from paramopt.acquisitions import UCB
from paramopt.optimizers.sklearn import BayesianOptimizer

workdir = pathlib.Path.cwd()
```

###  <li>Defining Target Parameters</li>

<img src=https://user-images.githubusercontent.com/88641432/179461813-4ba5c30b-3fba-40fb-ba8e-f7009ff24a1b.png width="50%">

Define parameters to be adjusted.
```python
param1 = ProcessParameter(name="Heating Temperature", values=[150, 180, 210, 230, 250])
param2 = ProcessParameter(name="Heating Time", values=[10, 20, 40, 80, 150, 220])
```
`name` is the parameter name and `values` is a list of possible values of the parameter.

Then, define a exploration space consisting of the parameters.
```python
space = ExplorationSpace([param1, param2])
```

This definition can be exported to / imported from a json file.
```python
space.to_json(workdir)  # export
space = ExplorationSpace.from_json(workdir)  # import
```

### <li>Creating Dataset</li>

<img src=https://user-images.githubusercontent.com/88641432/179461866-07f71fa1-0955-4980-8bbf-e70f7aa45581.png width="45%">

Create a dataset consisting of an explanatory variables with `X_names` and objective variables with `Y_names`.
```python
dataset = Dataset(X_names=space.names, Y_names="Evaluation")
```
Basically, X_names is passed the parameter namew, and Y_names is passed the name of the evaluations.
The dataset is managed by the `BayesianOptimizer` class described below.

This data can be exported to / imported from a csv file.
```python
dataset.to_csv(workdir)    # export
dataset = Dataset.from_csv(workdir)# import
```

### <li>Preparing GPR Model and Acquisition Function</li>
Use `sklearn.gaussian_process` for the GPR model.
Acquisition functions are provided in this package.
```python
model = GaussianProcessRegressor(kernel=RBF(10), normalize_y=True)
acquisition = UCB(1.0)
```

### <li>Optimizing Parameters</li>
Let's optimize parameters in the bayesian optimization loop.
Here is a function that simulates an experiment and returns an evaluation value for a given parameter combination.
```python
def experiment(x1, x2):
    return x1*np.sin(x1*0.05+np.pi) + x2*np.cos(x2*0.05)
```

The optimization flow is as follows:
```python
# Define optimizer
bo = BayesianOptimizer(
    workdir=Path.cwd(), exploration_space=space, dataset=dataset, model=model,
    acquisition=acquisition, objective_fn=experiment, random_seed=71)

# For max iterations:
for i in range(15):
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

### <li>GPR with Hyperparameter Auto-adjustment Ability</li>

<img src=https://user-images.githubusercontent.com/88641432/177728148-57ed7d52-07ec-4c5c-af1c-81afb7440860.png width="65%">

Sometimes GPR does not predict well like this:

<img src=https://user-images.githubusercontent.com/88641432/177728843-dea8cacb-60e5-4fbb-adf1-edeb894ccdde.png width="40%">

In this case, let's replace it with a model that automatically adjusts the hyperparameters.

```python
from paramopt.extensions import AutoHPGPR

def gpr_generator(exp, nro):
    return GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=(10**-exp, 10**exp)) \
                * ConstantKernel() \
                + WhiteKernel(),
        normalize_y=True, n_restarts_optimizer=nro)

model = AutoHPGPR(
    workdir=Path.cwd(), exploration_space=space, gpr_generator=gpr_generator,
    exp=list(range(1, 6)), nro=list(range(0, 10)))
```

The result is

<img src=https://user-images.githubusercontent.com/88641432/177729186-7dfe1249-8a2c-4ce7-9ec8-393e2b682970.png width="40%">

### <li>GIF Creation from Plot pngs</li>
Create a GIF movie from the obtained plot images

```python
paths = select_images()  # Opens a GUI dialog
create_gif(paths)
```
<img src="https://user-images.githubusercontent.com/88641432/177729552-23194201-8241-4c3f-b814-68e5bd69b4bb.PNG" width="30%"><img src=https://user-images.githubusercontent.com/88641432/177729289-6ab150dd-c487-488f-bb82-d52e94fb77e9.gif width="60%">


# License
MIT license.
