import os
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


def load_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    os.remove(temp_path)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks


def create_vectorstore(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore


def retrieve_documents(vectorstore, question):

    retrieved_docs = vectorstore.similarity_search(
        question,
        k=3
    )

    return retrieved_docs


def generate_answer(question, retrieved_docs):

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template= """
You are a document question-answering assistant.

Answer ONLY using the information in the context below.

If the answer is not present in the context, reply exactly:

"I couldn't find the answer in the uploaded document."

Do not use your own knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
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