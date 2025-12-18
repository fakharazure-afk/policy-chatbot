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

def retrieve_context(query, k=2, max_chars=900):
    q_emb = model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, k)
    context = ""
    for idx in I[0]:
        text = metadata[idx]["text"]
        if len(context) + len(text) > max_chars:
            break
        context += text + "\n"
    return context.strip()

question = st.text_input("Ask a question from policy:")

if question:
    context = retrieve_context(question)

    if not context:
        st.warning("No relevant policy text found.")
    else:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a policy assistant. "
                        "Answer strictly from the policy text provided below. "
                        "If the answer is not present, say you don't know.\n\n"
                        f"POLICY TEXT:\n{context}"
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0,
            max_tokens=250
        )

        st.write(response.choices[0].message.content)
