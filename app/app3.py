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

# TODO:モデル構築及びプロットに用いるデータの種別を指定（コア数、問題サイズ（初期変数））

# TODO:モデルの選択（線形、反比例、対数、線形飽和...）

# TODO:グラフのプロット（X軸対数化、Y軸対数化、プロット）
