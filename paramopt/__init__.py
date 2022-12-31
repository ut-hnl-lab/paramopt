from .acquisitions import EI, UCB, BaseAcquisition
from .extensions import (AutoHyperparameter, AutoHyperparameterRegressor,
                         create_gif, select_images)
from .optimizer import BayesianOptimizer
from .parameter import ExplorationSpace

__version__ = '2.0.0'
