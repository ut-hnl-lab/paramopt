from itertools import product

from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from paramopt.acquisitions.ucb import UCB
from paramopt.graphs.distribution import _plot_process_1d, _plot_process_2d
from paramopt.graphs.transition import _plot_transition


def f1(x):
    return x * np.sin(x)


def f2(x1, x2):
    return (x1 * np.sin(x1*0.05+np.pi) + x2 * np.cos(x2*0.05)) * 0.1


def make_train_dataset(X, y):
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    return X_train, y_train


def predict_with_gp(X_train, y_train, X):
    kernel = 1 * RBF(length_scale=10.0, length_scale_bounds='fixed')
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)

    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    ucb = UCB(c=1.0)
    acquisition = ucb(mean_prediction, std_prediction, X_train, y_train)
    next_X = X[np.argmax(acquisition)]

    return mean_prediction, std_prediction, acquisition, next_X


def test_dist_1d():
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    y = np.squeeze(f1(X))
    X_train, y_train = make_train_dataset(X, y)
    mean_prediction, std_prediction, acquisition, next_X = predict_with_gp(X_train, y_train, X)

    fig = _plot_process_1d(plt.figure(), X_train, y_train, X)
    fig.tight_layout()
    plt.show()
    fig = _plot_process_1d(plt.figure(), X_train, y_train, X, mean_prediction, std_prediction)
    fig.tight_layout()
    plt.show()
    fig = _plot_process_1d(plt.figure(), X_train, y_train, X, mean_prediction, std_prediction, acquisition, next_X, f1)
    fig.tight_layout()
    plt.show()


def test_dist_2d():
    X1 = np.linspace(start=0, stop=10, num=100).reshape(-1, 1)
    X2 = np.linspace(start=20, stop=50, num=100).reshape(-1, 1)
    X = np.array(list(product(X1.flatten(), X2.flatten())))
    y = np.squeeze(f2(X[:,0], X[:,1]))
    X_train, y_train = make_train_dataset(X, y)
    mean_prediction, std_prediction, acquisition, next_X = predict_with_gp(X_train, y_train, X)

    fig = _plot_process_2d(plt.figure(), X_train, y_train, [X1, X2])
    fig.tight_layout()
    plt.show()
    fig = _plot_process_2d(plt.figure(), X_train, y_train, [X1, X2], mean_prediction)
    fig.tight_layout()
    plt.show()
    fig = _plot_process_2d(plt.figure(), X_train, y_train, [X1, X2], mean_prediction, acquisition, next_X, f2)
    fig.tight_layout()
    plt.show()


def test_transition_nd():
    X1 = np.linspace(start=0, stop=10, num=100)
    X2 = np.linspace(start=200, stop=500, num=100)
    X3 = np.linspace(start=0, stop=-10, num=100)
    X4 = np.linspace(start=-20, stop=20, num=100)
    y = np.linspace(start=80, stop=100, num=100)
    X_train = np.vstack((
        np.random.choice(X1, 100),
        np.random.choice(X2, 100),
        np.random.choice(X3, 100),
        np.random.choice(X4, 100),
    )).T
    y_train = np.atleast_2d(np.random.choice(y, 100)).T

    fig = _plot_transition(plt.figure(), [X1,X2,X3,X4], X_train, y_train, ["pp1","pp2","pp3","pp4"], ["y"], Y_bounds=(0,100))
    fig.tight_layout()
    plt.show()
