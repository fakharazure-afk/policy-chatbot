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

def clean_text(text):
    # Remove null bytes and invalid characters
    text = text.replace("\x00", "")
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text

def retrieve_context(query):
    q_emb = model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, 1)  # get only ONE best chunk
    return metadata[I[0][0]]["text"].strip()

    q_emb = model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, k)
    context = ""
    for idx in I[0]:
        chunk = clean_text(metadata[idx]["text"])
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n"
    return context.strip()

question = st.text_input("Ask a question from policy:")

if question:
    context = retrieve_context(question)

    if not context:
        st.warning("No relevant policy text found.")
    else:
        try:
           response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "system",
            "content": "Answer strictly from the given policy text. If not found, say so."
        },
        {
            "role": "user",
            "content": context + "\n\nQuestion: " + question
        }
    ],
    temperature=0,
    max_tokens=150
)

            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error("Model rejected the request. Try a simpler question.")
