import streamlit as st
import subprocess
import pandas as pd

# DataFrameを生成するcache機能付の関数を定義
@st.cache
def get_df(s):
    return pd.DataFrame({'id':['1','2','3'], 'name':['X', 'Y', s]})

@st.cache
def gen_lib():
    subprocess.run(["jupyter", "nbconvert", "--to", "python", "lib.ipynb"])
# ライブラリノートを実行可能な形式に変換
gen_lib()

# text formを生成
s = st.text_input('input s')
df = get_df(s)

# DataFrameをテーブルで表示
st.markdown('# Table')
st.table(df)

# histgramを表示
st.markdown('# Histgram')
df['name'].hist()
st.pyplot()
