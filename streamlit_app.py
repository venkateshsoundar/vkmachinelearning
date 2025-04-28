import streamlit as st
import pandas as pd

st.title('🤖 This is the Machine Learning App')

st.write('This is the app build a machine learning model')
df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")

with st.expander('Data'):
  st.write("**Raw Data**")
  df

  st.write("**X**")
  X=df.drop('species',axis=1)
  X

  st.write("**y**")
  y=df.species
  y

with st.expander("**Visualizations**"):
  st.scatter_chart(data=df,x="bill_length_mm",y="body_mass_g",color="species")

#Data Preparation
with st.sidebar:
    st.header("**Input Features**")
    island = st.selectbox("Island", df["island"].unique())    
    bill_length_mm=st.slider('bill_length_mm',df['bill_length_mm'].min(),df['bill_length_mm'].max(),df['bill_length_mm'].mean())
    bill_depth_mm=st.slider('bill_depth_mm',df['bill_depth_mm'].min(),df['bill_depth_mm'].max(),df['bill_depth_mm'].mean())
    flipper_length_mm=st.slider('flipper_length_mm',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()),float(df['flipper_length_mm'].mean()))
    body_mass_g=st.slider('body_mass_g',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()),float(df['body_mass_g'].mean()))
    sex=st.selectbox("Sex", df["sex"].unique())
  
    data={
      'island':island,
      'bill_length_mm':bill_length_mm,
      'bill_depth_mm':bill_depth_mm,
      'flipper_length_mm':flipper_length_mm,
      'body_mass_g':body_mass_g,
      'sex':sex
    }

    input_df=pd.DataFrame(data,index=[0])

with st.expander('Input_Features'):
  input_df    
    
  
  
