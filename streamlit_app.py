import streamlit as st
import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from sklearn.ensemble import RandomForestClassifier
import os

# --- Streamlit page configuration ---
st.set_page_config(
    page_title="Custom Dataset QA Chatbot & Penguin Classifiers",
    layout="wide"
)
st.title('🤖 Custom Dataset QA Chatbot & Penguin Classifier')

# --- Configuration ---
st.sidebar.header("📂 Dataset Upload & Settings")
dataset_file = st.sidebar.file_uploader("Upload a CSV for QA", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv"
    )

df = load_data(dataset_file)

# Setup Langchain Chat Model (replace with your own API keys if needed)
deepseek_api_key = st.secrets.get('DEEPSEEK_API_KEY', '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')
model_name = st.secrets.get('MODEL_NAME', 'deepseek/deepseek-r1-zero:free')
openai_api_base = st.secrets.get('OPENAI_API_BASE', 'https://openrouter.ai/api')

llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=deepseek_api_key,
    openai_api_base=openai_api_base,
    temperature=0
)

# Create Pandas DataFrame agent
pandas_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# --- Chatbot ---
if 'qa_history' not in st.session_state:
    st.session_state['qa_history'] = []

st.header("💬 Ask Questions About the Uploaded Dataset")
for message in st.session_state['qa_history']:
    if message['role'] == 'user':
        st.chat_message("user").write(message['content'])
    else:
        st.chat_message("assistant").write(message['content'])

user_input = st.chat_input("Ask me anything about the dataset:")

if user_input:
    st.session_state['qa_history'].append({'role': 'user', 'content': user_input})
    st.chat_message("user").write(user_input)
    try:
        response = pandas_agent.run(user_input)
        st.session_state['qa_history'].append({'role': 'assistant', 'content': response})
        st.chat_message("assistant").write(response)
    except Exception as e:
        error_msg = "❌ No data available for that query or error occurred."
        st.session_state['qa_history'].append({'role': 'assistant', 'content': error_msg})
        st.chat_message("assistant").write(error_msg)

# --- Penguin Species Classifier (Original Section) ---
st.markdown("---")
st.header("🐧 Penguin Species Classifier")

with st.expander('Data & Visualization'):
    st.write("**Raw Data**")
    st.dataframe(df)
    if 'species' in df.columns:
        st.write("**Scatter Plot**")
        st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", color="species")

with st.sidebar:
    if 'species' in df.columns:
        st.header("🛠 Input Features for Prediction")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        input_data = {}
        for col in cat_cols:
            input_data[col] = [st.selectbox(col, df[col].unique())]
        for col in numeric_cols:
            input_data[col] = [st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )]
        input_df = pd.DataFrame(input_data)

        # Prepare for model
        X_raw = df.drop('species', axis=1)
        combined = pd.concat([input_df, X_raw], axis=0)
        encoded = pd.get_dummies(combined, columns=cat_cols)
        input_row = encoded.iloc[:1, :]
        X = encoded.iloc[1:, :]
        y = df['species'].map({v: i for i, v in enumerate(df['species'].unique())})

        # Train and predict
        clf = RandomForestClassifier()
        clf.fit(X, y)
        pred = clf.predict(input_row)
        proba = clf.predict_proba(input_row)
        proba_df = pd.DataFrame(proba, columns=df['species'].unique())

        with st.expander('Prediction Probabilities'):
            st.dataframe(proba_df)
        with st.expander('Predicted Species'):
            species_list = df['species'].unique()
            st.success(f"Predicted: {species_list[pred][0]}", icon="👍")
