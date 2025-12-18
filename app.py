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

def retrieve(query, k=4):
    q_emb = model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, k)
    return [metadata[i]["text"] for i in I[0]]

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
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
        temperature=0
    )
    st.write(response.output_text)
