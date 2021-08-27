import streamlit as st
import subprocess
import pandas as pd

dimension = st.sidebar.selectbox('プロットする次元',('２次元', '３次元'))

if dimension == '２次元':       # 2次元グラフの描画
    pass

elif dimension == '３次元':     # 3次元グラフの描画
    pass

else:
    pass
