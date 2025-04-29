import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="deepseek/deepseek-r1:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)

st.title('🤖 This is the Machine Learning App')

st.write('This is the app build a machine learning model')
df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")

with st.expander('Data'):
  st.write("**Raw Data**")
  df

  st.write("**X**")
  X_raw=df.drop('species',axis=1)
  X_raw

  st.write("**y**")
  y_raw=df.species
  y_raw

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
    penguins_df=pd.concat([input_df,X_raw],axis=0)
  

with st.expander('Input_Features'):
  st.write("**Input_Features**")
  input_df
  st.write("**Combined_Features**")
  penguins_df 

# Encode
with st.expander('Encoded_Features'):  
  encode=['island','sex']
  df_penguins = pd.get_dummies(penguins_df, columns=encode, prefix=encode)
  X=df_penguins[1:]
  inputrow=df_penguins[:1]

#Encode y
target_mapper={
  'Adelie':0,'Chinstrap':1,'Gentoo':2  
}
def target_encode(val):
  return target_mapper[val]

y=y_raw.apply(target_encode)

with st.expander('Data Preparation'):
  st.write('**Encoded X(input penguin)**')
  inputrow
  st.write('**Encoded y**')
  y

with st.expander('Prediction_Probablities'):
#Model Training
  clf=RandomForestClassifier()
  clf.fit(X,y)
  
  #Apply prediction
  predicition=clf.predict(inputrow)
  prediction_proba=clf.predict_proba(inputrow)
  prediction_proba_df = pd.DataFrame(prediction_proba)
  prediction_proba_df.rename(columns={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}, inplace=True)
  prediction_proba_df


with st.expander('Predicted_Species'):
  st.dataframe(
    prediction_proba_df,
    column_config={
        "Adelie": st.column_config.ProgressColumn(
            "Adelie",
            format="%f",
            width='medium',
            min_value=0,
            max_value=1,
        ),
        "Chinstrap": st.column_config.ProgressColumn(
            "Chinstrap",
            format="%f",
            width='medium',
            min_value=0,
            max_value=1,
        ),
        "Gentoo": st.column_config.ProgressColumn(
            "Gentoo",
            format="%f",
            width='medium',
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True,
      )

penguin_species=np.array(['Adelie','Chinstrap','Gentoo'])
st.success(penguin_species[predicition][0], icon="👍")

  
