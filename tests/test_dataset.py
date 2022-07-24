from pathlib import Path

import numpy as np
import pytest

from paramopt.structures.dataset import Dataset


# 1次元X，1次元Y格納
def test_add_x1d_y1d():
    d = Dataset("X1", "Y1")
    assert d.X_names == ["X1"]
    assert d.Y_names == ["Y1"]
    assert (d.X == np.array([[]])).all()
    assert (d.Y == np.array([[]])).all()
    assert d.dimension_X == 1
    assert d.dimension_Y == 1
    with pytest.raises(ValueError):
        d.last_label

    d = d.add(1, 1, "#1")
    assert d.X_names == ["X1"]
    assert d.Y_names == ["Y1"]
    assert (d.X == np.array([[1]])).all()
    assert (d.Y == np.array([[1]])).all()
    assert d.dimension_X == 1
    assert d.dimension_Y == 1
    assert d.last_label == "#1"

    with pytest.raises(ValueError):
        Dataset("x", "y", X=[1,2], Y=[3])

    with pytest.raises(ValueError):
        Dataset("x", "y", X=[1], Y=[2,3])

    with pytest.raises(ValueError):
        d.add([1, 2], 1, "#2")

    with pytest.raises(ValueError):
        d.add(1, [1, 2], "#2")

    with pytest.raises(ValueError):
        d.add([[1]], [[1], [2]], "#2")


# 2次元X，2次元Y格納
def test_add_x2d_y2d():
    d = Dataset(["X1", "X2"], ["Y1", "Y2"])
    assert d.X_names == ["X1", "X2"]
    assert d.Y_names == ["Y1", "Y2"]
    assert d.X.shape == (0,2)
    assert d.Y.shape == (0,2)
    assert d.dimension_X == 2
    assert d.dimension_Y == 2
    with pytest.raises(ValueError):
        d.last_label

    d = d.add([1, -1], [3, -3], "#1")
    d = d.add([2, -2], [4, -4], "#1")
    d = d.add([1, -1], [3, -3], "#2")
    assert d.X_names == ["X1", "X2"]
    assert d.Y_names == ["Y1", "Y2"]
    assert (d.X == np.array([[1,-1],[2,-2],[1,-1]])).all()
    assert (d.Y == np.array([[3,-3],[4,-4],[3,-3]])).all()
    assert d.dimension_X == 2
    assert d.dimension_Y == 2
    assert d.last_label == "#2"


# ファイル入出力
def test_file_io():
    path = Path.cwd()/'tests'/'output'/'dataset_file_io'
    d1 = Dataset(["X1", "X2"], ["Y1", "Y2"])
    d1.to_csv(path)
    d2 = Dataset.from_csv(path, 2, 2)
    assert d2.X_names == ["X1", "X2"]
    assert d2.Y_names == ["Y1", "Y2"]
    assert d2.X.shape == (0,2)
    assert d2.Y.shape == (0,2)
    assert d2.dimension_X == 2
    assert d2.dimension_Y == 2

    d3 = d2.add([9,8], [7,6], "#1").add([5,4], [3,2], "#2")
    d3.to_csv(path/'custom_name.csv')
    assert d3.X.shape == (2,2)
    assert d3.Y.shape == (2,2)
    assert len(d3) == 2

    d4 = Dataset.from_csv(path/'custom_name.csv', 2, 2)

    with pytest.raises(ValueError):
        Dataset.from_csv(path/'custom_name.csv', 1, 1)


# 複数データの追加
def test_add_multidata():
    d1 = Dataset(["X1", "X2"], "Y1", X=[0,0], Y=[0], labels=["init"])
    d2 = d1.add([[1,2], [-1,-2]], [[3], [4]])
    assert d2.labels == ["init", "", ""]
    d3 = d1.add([[1,2], [-1,-2]], [[3], [4]], ["a", "b"])
    assert d3.labels == ["init", "a", "b"]
    with pytest.raises(ValueError):
        d1.add([[1,2], [-1,-2]], [[3], [4]], ["a"])


def test_slice():
    d = Dataset("x", "y", X=[[1],[2],[3]], Y=[[4],[5],[6]])
    d2 = d[:2]
    assert (d2.X == np.array([[1],[2]])).all()
    assert (d2.Y == np.array([[4],[5]])).all()
