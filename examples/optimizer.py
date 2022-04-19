from typing import Callable
import numpy as np

from paramopt import BaseLearner


def search_1d(
    gpr: BaseLearner, f: Callable, prefix: str = '', noisy: bool = True
) -> None:
    gpr.add_parameter('param', np.arange(0, 10.5, 0.5))
    gpr.fit(0, f(0), tag=prefix+str(0))
    gpr.fit(1, f(1), tag=prefix+str(0))
    gpr.fit(10, f(10), tag=prefix+str(0))
    gpr.graph(objective_fn=f)

    for i in range(10):
        next_x, = gpr.next()
        y = f(next_x)
        if noisy:
            y += np.random.normal(scale=0.1)
        gpr.fit(next_x, y, tag=prefix+str(i+1))
        gpr.graph(objective_fn=f)


def search_2d(
    gpr: BaseLearner, f: Callable, prefix: str = '', noisy: bool = True
) -> None:
    gpr.add_parameter('param1', np.arange(180, 285, 5))
    gpr.add_parameter('param2', np.arange(10, 220, 10))
    gpr.fit((180, 10), f(180, 10), tag=prefix+str(0))
    gpr.fit((185, 15), f(185, 15), tag=prefix+str(0))
    gpr.fit((180, 210), f(180, 210), tag=prefix+str(0))
    gpr.fit((280, 10), f(280, 10), tag=prefix+str(0))
    gpr.fit((280, 210), f(280, 210), tag=prefix+str(0))
    gpr.graph(objective_fn=f)

    for i in range(15):
        next_x = gpr.next()
        y = f(*next_x)
        if noisy:
            y += np.random.normal(scale=1.0)
        gpr.fit(next_x, y, tag=prefix+str(i+1))
        gpr.graph(objective_fn=f)
