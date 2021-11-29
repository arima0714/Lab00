# import matplotlib.pyplot as plt
import streamlit as st
import subprocess
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px


def app():
    @st.cache
    def gen_lib():
        subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
        subprocess.run(["mv", "../lib/lib.py", "libLab00.py"])

    # TODO:元データとなるDFの指定

    ## ベンチマークの指定
    # 生データの取得
    benchmark_name = [
        st.selectbox(options=["cg", "ep", "ft", "is", "lu", "mg"], label="ベンチマーク名")
    ]

    ### 列の作成
    ## 問題サイズの指定(チェックボックス)
    column_for_core, column_for_size = st.columns(2)

    column_for_core.subheader("問題サイズの指定")

    enable_a = column_for_core.checkbox("問題サイズA", value=True)
    enable_b = column_for_core.checkbox("問題サイズB")
    enable_c = column_for_core.checkbox("問題サイズC")
    enable_d = column_for_core.checkbox("問題サイズD")
    enable_e = column_for_core.checkbox("問題サイズE")
    enable_f = column_for_core.checkbox("問題サイズF")

    program_size_set = set()
    if enable_a:
        program_size_set.add("A")
    if enable_b:
        program_size_set.add("B")
    if enable_c:
        program_size_set.add("C")
    if enable_d:
        program_size_set.add("D")
    if enable_e:
        program_size_set.add("E")
    if enable_f:
        program_size_set.add("F")

    program_size_list = sorted(list(program_size_set))
    st.write(program_size_list)

    ## コア数の指定(チェックボックス)
    column_for_core.subheader("コア数")
    #### 実際にデータ内にあるコア数のリストを作成
    enable001 = column_for_core.checkbox("コア数1", value=True)
    enable002 = column_for_core.checkbox("コア数2")
    enable004 = column_for_core.checkbox("コア数4")
    enable008 = column_for_core.checkbox("コア数8")
    enable016 = column_for_core.checkbox("コア数16")
    enable032 = column_for_core.checkbox("コア数32")
    enable064 = column_for_core.checkbox("コア数64")
    enable128 = column_for_core.checkbox("コア数128")
    enable256 = column_for_core.checkbox("コア数256")

    num_of_core_set = set()
    if enable001:
        num_of_core_set.add(1)
    if enable002:
        num_of_core_set.add(2)
    if enable004:
        num_of_core_set.add(4)
    if enable008:
        num_of_core_set.add(8)
    if enable016:
        num_of_core_set.add(16)
    if enable032:
        num_of_core_set.add(32)
    if enable064:
        num_of_core_set.add(64)
    if enable128:
        num_of_core_set.add(128)
    if enable256:
        num_of_core_set.add(256)
    num_of_core_list = sorted(list(num_of_core_set))
    st.write(num_of_core_list)

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
