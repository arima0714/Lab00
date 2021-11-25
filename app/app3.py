import matplotlib.pyplot as plt
import streamlit as st
import subprocess
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def app():
    @st.cache
    def gen_lib():
        subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
        subprocess.run(["mv", "../lib/lib.py", "libLab00.py"])

# TODO:元データとなるDFの指定

## TODO:ベンチマークの指定
## TODO:問題サイズの指定(チェックボックス)
## TODO:コア数の指定(チェックボックス)

# TODO:モデル構築及びプロットに用いるデータの種別を指定（コア数、問題サイズ（初期変数））
## TODO:元データとなるDFから列名を取得
## TODO:取得した列名をチェックボックス化して、チェックされた変数をリスト化
## TODO:リスト化された変数をモデルの構築に使用

# TODO:モデルの選択（線形、反比例、対数、線形飽和...）
## TODO:モデル名をチェックボックス化して、チェックされた変数をリスト化
## TODO:リスト化されたモデル名をモデルの構築に使用

# TODO:モデルの構築
## TODO:{モデル名:モデル}となるようにモデルを格納

# TODO:グラフのプロット（X軸対数化、Y軸対数化、プロット）
## TODO:元データのプロット
## TODO:元データの横軸最低値から横軸最大値でモデルを用いて予測
## TODO:モデルから予測されたデータをプロット
