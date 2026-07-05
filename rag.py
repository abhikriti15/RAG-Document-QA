import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

# Works both locally (.env) and on Streamlit Cloud (Secrets)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# Constants
# --------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.1-8b-instant"

# --------------------------------------------------
# Load PDF
# --------------------------------------------------

def load_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    os.remove(temp_path)

    return documents


# --------------------------------------------------
# Split Documents
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------

def create_vectorstore(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore


# --------------------------------------------------
# Retrieve Documents
# --------------------------------------------------

def retrieve_documents(vectorstore, question):

    return vectorstore.similarity_search(
        question,
        k=TOP_K
    )


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(question, retrieved_docs):

    if not GROQ_API_KEY:
        raise ValueError(
            "Groq API key not found. Please configure GROQ_API_KEY in Streamlit Secrets or your .env file."
        )

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a Retrieval-Augmented Generation (RAG) assistant.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context, reply exactly:

"I couldn't find the answer in the uploaded document."

Do not use outside knowledge.
Do not guess.
Do not fabricate information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content