import streamlit as st
import pandas as pd

st.title('🤖 This is the Machine Learning App')

st.write('This is the app build a machine learning model')
with st.expander('Data'):
  st.write("**Raw Data**")
  df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

