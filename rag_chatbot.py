import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI  
import os
# ==========================================
# CONFIGURATION

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================================
# STEP 0: SETUP THE DATASET

def create_dataset():
    data = [
        {"id": 1, "customer": "Amit", "product": "Laptop", "amount": 55000, "date": "2024-01-12"},
        {"id": 2, "customer": "Amit", "product": "Mouse", "amount": 700, "date": "2024-02-15"},
        {"id": 3, "customer": "Riya", "product": "Mobile", "amount": 30000, "date": "2024-01-05"},
        {"id": 4, "customer": "Riya", "product": "Earbuds", "amount": 1500, "date": "2024-02-20"},
        {"id": 5, "customer": "Karan", "product": "Keyboard", "amount": 1200, "date": "2024-03-01"}
    ]
    with open('transactions.json', 'w') as f:
        json.dump(data, f)
    print("Dataset 'transactions.json' created successfully.")

# ==========================================
# STEP 1: LOAD & PREPROCESS

def load_and_preprocess_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        desc = f"On {item['date']}, {item['customer']} purchased a {item['product']} for {item['amount']}."
        texts.append(desc)
    
    return texts

# ==========================================
# STEP 2: Creating Embeddings

def create_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Create embeddings using SentenceTransformer.
    Accepts an optional model_name so callers (like streamlit) can pass a model choice.
    """
    print("Generating embeddings... (this may take a moment)")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings, model


# ==========================================
# STEP 3: Similarity search 

def retrieve_transactions(query, embeddings, texts, model, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [texts[i] for i in top_indices]
    return results

# ==========================================
# STEP 4: Building the chatbot

def generate_answer(query, context_texts):
    context_block = "\n".join(context_texts)
    
    prompt = f"""
    You are a helpful retail assistant. 
    Answer the user's question based ONLY on the following context. 
    
    Context:
    {context_block}
    
    Question: 
    {query}
    """
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"

# ==========================================
# main part to run the chatbot in console

if __name__ == "__main__":
    create_dataset()
    texts = load_and_preprocess_data('transactions.json')
    embeddings, model = create_embeddings(texts)
    
    print("\n--- RAG Chatbot Ready! (Type 'exit' to quit) ---")
    
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        relevant_docs = retrieve_transactions(user_query, embeddings, texts, model, top_k=3)
        
        # Debugging: Print what we found 
        print(f"\n[System Log] Context Found: {relevant_docs}\n")
        
        answer = generate_answer(user_query, relevant_docs)
        print(f"Bot: {answer}")





