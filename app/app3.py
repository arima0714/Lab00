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
        