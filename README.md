📘 RAG Transaction Chatbot

A Retrieval-Augmented Generation (RAG) system for answering questions over transactional data.

🚀 Overview

This project implements a RAG-based chatbot capable of answering natural-language questions about a set of retail transactions.

The chatbot works by:

Retrieving relevant transactions using semantic search (embeddings + cosine similarity)

Augmenting an LLM with the retrieved context

Ensuring answers are grounded ONLY in the provided context (no hallucinations)

This architecture makes the chatbot reliable, explainable, and easy to extend.

✨ Features
🔹 Transaction Retrieval

Converts each transaction into a descriptive sentence

Uses SentenceTransformers (all-MiniLM-L6-v2) to generate embeddings

Retrieves top-k relevant transactions using cosine similarity

🔹 LLM-Powered Answers

Uses OpenAI’s GPT API (configurable)

The LLM answers based only on retrieved transaction context

If information is missing → responds with:

"I don't know based on the available transactions."

🔹 CLI Chatbot

Ask follow-up questions in a conversational loop

Example queries:

“Show me Riya’s purchase history”

“What is Amit’s total spending?”

“List all transactions from February”

🔹 Clean & Modular Code

Easy-to-read Python script (rag_chatbot.py)

Highly extendable

Designed exactly as required in the assignment specification

📂 Project Structure
.
├── rag_chatbot.py          # Main chatbot program
├── transactions.json       # Sample transaction dataset
├── README.md               # Project documentation
└── requirements.txt        # Required Python libraries

🛠️ Technologies Used
Component	Technology
Embeddings	SentenceTransformers (all-MiniLM-L6-v2)
LLM	OpenAI GPT models (default: gpt-4.1-mini)
Similarity Search	Cosine similarity on normalized vectors
Language	Python 3
📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2️⃣ Install Dependencies
pip install -r requirements.txt


Or manually:

pip install sentence-transformers numpy openai

3️⃣ Add Your API Key

Linux/macOS:

export OPENAI_API_KEY="your_api_key_here"


Windows CMD:

setx OPENAI_API_KEY "your_api_key_here"

▶️ Running the Chatbot

Simply execute:

python rag_chatbot.py


You will see:

RAG Transaction Chatbot
Type your question...


Example:

You: Show me Amit’s purchases
Bot: Amit purchased a Laptop for ₹55000 on 2024-01-12...

🧠 How It Works (Architecture)
🔹 1. Preprocessing

Each transaction → human-readable text:

On 2024-03-01, Karan purchased a Keyboard for ₹1200.

🔹 2. Embedding

Texts encoded using MiniLM → high-dimensional vectors.

🔹 3. Retrieval

Cosine similarity used to fetch top-k relevant transactions.

🔹 4. Generation

Retrieved context + user question → LLM → final grounded answer.

📊 Example Transaction Queries
User Question	Chatbot Capability
“Show me Riya’s history”	Lists Riya’s purchases
“What is Amit’s total spending?”	Sums his transactions
“Give me Feb transactions”	Filters by date
“Which product was purchased most?”	Context-based reasoning

Add conversation memory

🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgments

Built as part of an assignment to demonstrate practical understanding of RAG systems, embeddings, retrieval pipelines, and LLM grounding techniques.