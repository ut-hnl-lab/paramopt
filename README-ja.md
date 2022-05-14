【[English](https://github.com/ut-hnl-lab/paramopt/blob/main/README.md)】

# ParamOpt
scikit-learn のガウス過程回帰をラップした, ベイズ最適化ライブラリ. (GpyOptにも対応しているが, サポート終了のため非推奨.)

## Description
ガウス過程回帰モデルを利用し, プロセスパラメータの最適化を行う. 手順は以下の通り.
1. カーネルなどの項目を設定してモデルをインスタンス化する.
2. プロセスパラメータを追加する.
3. 次の探索パラメータを取得する.
4. 取得したパラメータで実験し, 評価値を得る.
5. 評価値をモデルに学習させる.
6. 3~5を繰り返す.

学習の中断・再開や, 学習中の様子のグラフ化,・保存・gif動画化もサポート.

## Examples
1次元のシンプルな探索を行う例.<br>
詳しくは[examples](https://github.com/ut-hnl-lab/paramopt/tree/main/examples)を参照.

```python
from sklearn.gaussian_process.kernels import *
from paramopt.acquisitions import UCB
from paramopt.optimizers.sklearn import BayesianOptimizer

bo = BayesianOptimizer(  # 1
    savedir='tests',
    kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
    acqfunc=UCB(c=2.0),
    random_seed=71)

bo.add_parameter(name='parameter', space=range(10))  # 2

for i in range(10):  # 6
    next_x, = bo.next()  # 3
    y = {The score of the experimental result with "next_x" parameters}  # 4
    bo.fit(next_x, y, label=i+1)  # 5
    bo.plot()
```

gif動画の生成.
```python
from paramopt import select_images, create_gif

paths = select_images()
create_gif(paths)
```

## Demo
sin, cosで構成した関数にモデルをフィッティングさせる.

［凡例］
* グレーの線: 目的関数. 通常はこの分布が不明なため, データ駆動的に調べる.
* 黒点: 入力したデータ. 赤星は最新の値.
* 青線: データを学習した後のモデルの予測分布.
* 赤線(1D) or コンター図(2D: 獲得関数値.

|1次元パラメータの探索|2次元パラメータの探索|
|---|---|
|<img src="https://user-images.githubusercontent.com/88641432/163951938-5363d08b-15aa-436e-bccc-044dc771be80.gif" height=250>|<img src="https://user-images.githubusercontent.com/88641432/163952263-5861449f-5057-49a8-96e4-8c8f7e735a7c.gif" height=300>|

## Installation
```
pip install git+https://github.com/ut-hnl-lab/paramopt.git
```

## Requirement
* Python 3.6+
* numpy
* pandas
* pillow
* scikit-learn
* matplotlib

\[Optional\]
* gpy
* gpyopt
* natsort
