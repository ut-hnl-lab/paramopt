from itertools import product
from pathlib import Path

import numpy as np
import pytest

from paramopt.structures import ExplorationSpace, ProcessParameter


# 1次元空間格納
def test_add_1d():
    ProcessParameter.N_GRID_SPLITS = 5
    pp = ProcessParameter("first", [0,1,2])
    es = ExplorationSpace([pp])

    assert es.dimension == 1
    assert es.names == [pp.name]
    print(es.spaces)
    print([np.array(pp.values)])
    assert (es.spaces[0] == np.array(pp.values)).all()
    assert (es.grid_spaces[0] == np.array(pp.grid_values)).all()
    assert (es.conbinations() == np.array([0,1,2]).reshape(-1,1)).all()
    assert (es.grid_conbinations() == np.linspace(0,2,5).reshape(-1,1)).all()

# 2次元空間格納
def test_add_2d():
    ProcessParameter.N_GRID_SPLITS = 5
    pp1 = ProcessParameter("first", [0,1,2])
    pp2 = ProcessParameter("second", [7,10])
    es = ExplorationSpace([pp1, pp2])

    assert es.dimension == 2
    assert es.names == [pp1.name, pp2.name]
    assert (es.spaces[0] == np.array(pp1.values)).all()
    assert (es.spaces[1] == np.array(pp2.values)).all()
    assert (es.grid_spaces[0] == np.array(pp1.grid_values)).all()
    assert (es.grid_spaces[1] == np.array(pp2.grid_values)).all()
    assert (es.conbinations() == np.array(list(product(np.array([0,1,2]), np.array([7,10])))).reshape(-1,2)).all()
    assert (es.grid_conbinations() == np.array(list(product(np.linspace(0,2,5), np.linspace(7,10,5)))).reshape(-1,2)).all()

# 重複空間名入力 -> エラー
def test_name_dup():
    pp1 = ProcessParameter("first", np.array([0,1,2]))
    pp2 = ProcessParameter("first", np.array([7,10]))
    with pytest.raises(ValueError):
        ExplorationSpace([pp1, pp2])

# ファイル入出力
def test_file_io():
    path = Path.cwd()/'tests'/'output'/'space_file_io'
    path.mkdir(exist_ok=True, parents=True)
    pp1 = ProcessParameter("first", [0,1,2])
    pp2 = ProcessParameter("second", [7,10])
    es1 = ExplorationSpace([pp1, pp2])
    es1.to_json(path)

    es2 = ExplorationSpace.from_json(path/ExplorationSpace.EXPORT_NAME)

    assert es1 == es2
