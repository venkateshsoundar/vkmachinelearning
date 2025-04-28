import streamlit as st
import pandas as pd

st.title('🤖 This is the Machine Learning App')

st.write('This is the app build a machine learning model')
df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
df

