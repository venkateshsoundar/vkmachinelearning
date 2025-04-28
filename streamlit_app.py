import streamlit as st
import pandas as pd

st.title('🤖 This is the Machine Learning App')

st.write('This is the app build a machine learning model')
with st.expander('Data'):
  st.write("**Raw Data**")
  df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write("**X**")
  X=df.drop('species',axis=1)
  X

  st.write("**y**")
  y=df.species
  y

with st.expander("**Visualizations**"):
  st.scatter_chart(data=df,x="bill_length_mm",y="body_mass_g",color="species")
