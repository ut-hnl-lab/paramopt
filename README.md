【[日本語](https://github.com/ut-hnl-lab/paramopt/blob/main/README-ja.md)】

# ParamOpt
A library for Bayesian optimization, which wraps scikit-learn's Gaussian process regression. (GpyOpt is also available, but is deprecated since it is no longer supported.)

## Description
You can optimize process parameters by using Gaussian process regression model. The procedure is as follows:
1. Instantiate the model specifying arguments such as kernel.
2. Register process parameters.
3. Get the next parameters.
4. Experiment with the obtained parameters and score the results.
5. Train the model with the score.
6. Repeat 3-5.

This library also supports pausing and resuming learning, graphing, saving, and creating gif movies of the learning process..

## Examples
Example of a simple 1D exploration.<br>
See [examples](https://github.com/ut-hnl-lab/paramopt/tree/main/examples) for more details.

```python
from sklearn.gaussian_process.kernels import *
from paramopt import GPR, UCB

gpr = GPR(  # 1
    savedir='tests',
    kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
    acqfunc=UCB(c=2.0),
    random_seed=71)

gpr.add_parameter(name='parameter', space=range(10))  # 2

for i in range(10):  # 6
    next_x, = gpr.next()  # 3
    y = [The score of the experimental result with "next_x" parameters]  # 4
    gpr.fit(next_x, y, tag=i+1)  # 5
    gpr.graph()
```

Creation of a GIF animation.
```python
from paramopt import select_images, create_gif

paths = select_images()
create_gif(paths)
```

## Demo
Model fitting to a function composed of sin and cos.

\[Legend\]
* Gray line: Objective function. Since this distribution is usually unknown, it is studied in a data-driven way.
* Black dots: Input data. The red star means its latest data.
* Blue line: Predicted distribution by the model after learning the data.
* Red line(1D) or contour map(2D): Acquisition function values.

|1D parameter exploration|2D parameter exploration|
|---|---|
|<img src="https://user-images.githubusercontent.com/88641432/163951938-5363d08b-15aa-436e-bccc-044dc771be80.gif" height=250>|<img src="https://user-images.githubusercontent.com/88641432/163952263-5861449f-5057-49a8-96e4-8c8f7e735a7c.gif" height=300>|

## Installation
```
pip install git+https://github.com/ut-hnl-lab/paramopt.git
```

## Requirement
* Python 3.6+
* gpy
* gpyopt
* matplotlib
* natsort
* numpy
* pandas
* pillow
* scikit-learn
