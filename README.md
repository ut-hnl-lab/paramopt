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

学習の中断・再開や, 学習中の様子のグラフ化,・保存もサポート.

## Demo
sin, cosで構成した関数にモデルをフィッティングさせる.

［凡例］
* グレーの線：目的関数. 通常はこの分布が不明なため, データ駆動的に調べる.
* 黒点：入力したデータ. 赤星は最新の値.
* 青線：データを学習した後のモデルの予測分布.
* 赤線 or コンター図：獲得関数値.

➀ 1次元パラメータの探索

<img src="https://user-images.githubusercontent.com/88641432/163951938-5363d08b-15aa-436e-bccc-044dc771be80.gif" height=250>

➁ 2次元パラメータの探索

<img src="https://user-images.githubusercontent.com/88641432/163952263-5861449f-5057-49a8-96e4-8c8f7e735a7c.gif" height=300>

## Examples
1次元のシンプルな探索を行う例.

```python
from sklearn.gaussian_process.kernels import *
from paramopt import GPR, UCB

gpr = GPR(  # 1
    savedir='tests',
    kernel=RBF(length_scale=0.5) * ConstantKernel() + WhiteKernel(),
    acqfunc=UCB(c=2.0),
    random_seed=71)

gpr.add_parameter('parameter', range(10))  # 2

for i in range(10):
    next_x, = gpr.next()  # 3
    y = [next_xパラメータで実験をした結果の評価値]  # 4
    gpr.fit(next_x, y, tag=i+1)  # 5
    gpr.graph()
```

## Installation
```
git clone https://github.com/ut-hnl-lab/paramopt.git
pip install .\paramopt
```
