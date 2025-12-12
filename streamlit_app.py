
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# Import rag utilities
from rag_chatbot import (
    create_dataset,
    load_and_preprocess_data as build_texts_from_json,
    create_embeddings,
    retrieve_transactions,
    generate_answer as generate_answer_openai,
)


# App
# -----------------------------
st.set_page_config(page_title="RAG Chatbot UI", layout="wide")
st.title("RAG Chatbot ")

# # Sidebar
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_name = st.sidebar.selectbox("Sentence Transformer model", ["all-MiniLM-L6-v2"], index=0)

# Ensure dataset exists
DATA_FILE = "transactions.json"
if not os.path.exists(DATA_FILE):
    create_dataset(DATA_FILE)

# Load transactions into dataframe for charts
with open(DATA_FILE, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

# Build texts and embeddings
texts = build_texts_from_json(DATA_FILE)
embeddings, s_model = create_embeddings(texts, model_name)

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Chat")
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
        st.session_state.last_answer = None

    user_input = st.text_input("Ask a question about transactions:")
    if st.button("Send") and user_input:
        relevant_docs = retrieve_transactions(user_input, embeddings, texts, s_model, top_k=3)
        st.write("**[System Log] Context Found:**")
        for doc in relevant_docs:
            st.write(f"- {doc}")

        if not api_key:
            st.error("Please provide your OpenAI API key in the sidebar to generate answers.")
        else:
            with st.spinner("Generating answer from OpenAI..."):
                answer = generate_answer_openai(api_key, user_input, relevant_docs)

            st.session_state.last_question = user_input
            st.session_state.last_answer = answer

            st.markdown("**Bot:**")
            st.write(answer)

    st.markdown("---")
    if st.button("Show my last question"):
        if st.session_state.last_question:
            st.info(f"Last question: {st.session_state.last_question}")
            st.write("Last answer:")
            st.write(st.session_state.last_answer)
        else:
            st.warning("No previous question found in this session.")

with col2:
    st.subheader("Data & Charts")
    st.dataframe(df)

    df_month = df.copy()
    df_month["month"] = df_month["date"].dt.to_period("M").dt.to_timestamp()
    spend_per_month = df_month.groupby("month")["amount"].sum().reset_index()

    st.markdown("**Spend per month**")
    if not spend_per_month.empty:
        spend_per_month = spend_per_month.set_index("month")
        st.bar_chart(spend_per_month)
        st.table(spend_per_month)
    else:
        st.write("No transactions to show.")

 


