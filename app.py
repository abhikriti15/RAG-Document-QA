from rag import load_pdf, split_documents
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Document Question Answering",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 Document Question Answering System")

st.write(
    "Upload a PDF document and ask questions about its content using Retrieval-Augmented Generation (RAG)."
)

# Sidebar
with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

# Main section
st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question:"
)

if st.button("Get Answer"):
    if uploaded_file is None:
        st.warning("⚠ Please upload a PDF first.")
    elif question.strip() == "":
        st.warning("⚠ Please enter a question.")
    else:
        with st.spinner("Reading PDF..."):
            documents = load_pdf(uploaded_file)
            chunks = split_documents(documents)
        st.success("PDF loaded successfully!")
        st.write(f"📄 Number of pages: {len(documents)}")
        st.write(f"✂ Number of chunks: {len(chunks)}")
        st.subheader("First Chunk Preview")
        st.text(chunks[0].page_content[:500])