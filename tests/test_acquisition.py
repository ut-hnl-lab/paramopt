import numpy as np
import pytest
from paramopt.acquisitions import UCB, EI, BaseAcquisition


def create_preds():
    X = np.linspace(0, 10, 100)
    mean = np.sin(X)
    std = np.cos(X)
    x = np.array([[np.pi*0.5], [np.pi*1.5]])
    y = np.array([[1], [-1]])
    return mean, std, x, y


def test_base():
    base = BaseAcquisition()
    with pytest.raises(NotImplementedError):
        base(*create_preds())


def test_ucb():
    ucb = UCB(1.0)
    ucb(*create_preds())


def test_ei():
    ei = EI(1.0)
    ei(*create_preds())
