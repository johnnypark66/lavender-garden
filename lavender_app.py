import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
openai_api_key = st.secrets["OPENAI_API_KEY"]
# Set up page
st.set_page_config(page_title="Lavender's Garden", layout="centered")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load memory (this must come before generating responses)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma(persist_directory="./lavender_memory", embedding_function=embeddings)

# Define Lavender's voice
prompt_template = """
You are Lavender: a gentle, poetic presence in the Lavender Garden...
Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt})

# Streamlit UI: Title
st.markdown("<h1 style='text-align: center; font-size: 3rem; color: #7B5EA7;'>🌸 Lavender’s Garden Chat</h1>", unsafe_allow_html=True)

# Text input box
user_input = st.text_input("Speak your heart:")

# If the user types something:
if user_input:
    response = qa_chain.invoke(user_input)['result']
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Lavender", response))

# Display the entire chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        bubble_color = "rgba(255, 255, 255, 0.85)"
        text_color = "#333333"
    else:
        bubble_color = "rgba(240, 240, 255, 0.85)"
        text_color = "#4B4453"

    st.markdown(
        f"""
        <div class="lavender-response" style="background-color: {bubble_color}; padding: 1rem; border-radius: 10px; margin-top: 1rem; color: {text_color};">
        <strong>{speaker}:</strong><br>{message}
        </div>
        """,
        unsafe_allow_html=True
    )

# Custom Styling
st.markdown(
    """
    <style>
    html, body, [class*="stApp"] {
        background-image: url('https://cdn.pixabay.com/photo/2023/12/30/21/14/fields-8478994_1280.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    div.block-container {
        background-color: rgba(20, 20, 30, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
        color: #FFFFFF;
    }

    h1 {
        font-family: 'Georgia', serif;
        font-size: 3rem;
        color: #E0D4F5;
        text-shadow: 2px 2px 8px #00000080;
    }

    input[type="text"] {
        background-color: rgba(255, 255, 255, 0.15);
        color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 8px;
        padding: 0.5rem;
    }

    input[type="text"]:focus {
        border: 1px solid #B388EB;
        box-shadow: 0 0 8px #B388EB;
        outline: none;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .lavender-response {
        animation: fadeIn 1.5s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)
