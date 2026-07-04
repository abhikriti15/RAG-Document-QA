from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


def load_pdf(uploaded_file):
    """
    Load uploaded PDF and return LangChain documents.
    """

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Load the PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Delete temporary file
    os.remove(temp_path)

    return documents

def split_documents(documents):
    """
    Split documents into smaller chunks for retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks