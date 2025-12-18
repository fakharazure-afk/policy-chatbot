import streamlit as st
import os
import pickle
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Policy Chatbot")
st.title("Policy Chatbot")

@st.cache_resource
def load_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("policy_index.faiss")
    with open("policy_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

model, index, metadata = load_data()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def retrieve(query, k=2, max_chars=1200):
    q_emb = model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, k)
    collected = ""
    results = []
    for i in I[0]:
        if len(collected) >= max_chars:
            break
        text = metadata[i]["text"]
        collected += text
        results.append(text)
    return results

question = st.text_input("Ask a question from policy:")

if question:
    context = "\n".join(retrieve(question))
    prompt = f"""
Use only the policy context below.
If answer not found, say you don't know.

Policy:
{context}

Question: {question}
Answer:
"""

    response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": "You answer strictly from policy text."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    max_tokens=300
)
