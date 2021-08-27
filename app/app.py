import streamlit as st
import subprocess
import pandas as pd


@st.cache
def gen_lib():
    subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
    subprocess.run(["mv", "lib.py", "libLab00.py"])


# ライブラリノートを実行可能な形式に変換
gen_lib()
# 変換されたライブラリノートをimport
import libLab00 as lib

dimension = st.sidebar.selectbox("プロットする次元", ("２次元", "３次元"))

if dimension == "２次元":  # 2次元グラフの描画

    st.markdown("# ２次元グラフのプロット")

elif dimension == "３次元":  # 3次元グラフの描画

    st.markdown("# ３次元グラフのプロット")

else:
    pass
