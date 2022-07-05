import pytest

from paramopt.structures.parameter import ProcessParameter


# N要素リスト格納
def test_nd():
    name = "nd"
    values = [0,1,2]
    ProcessParameter.N_GRID_SPLITS = 5
    pp = ProcessParameter(name, values)

    assert pp.name == name
    assert pp.values == values
    assert pp.grid_values == [0., 0.5, 1., 1.5, 2.]


# 1要素リスト格納
def test_1d():
    name = "1d"
    values = [0]
    pp = ProcessParameter(name, values)

    assert pp.name == name
    assert pp.values == values
    assert pp.grid_values == [0]


# 空配列格納 -> エラー
def test_0d():
    name = "0d"
    values = []
    with pytest.raises(ValueError):
        pp = ProcessParameter(name, values)
