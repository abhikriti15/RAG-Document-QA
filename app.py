import streamlit as st
from rag import (
    load_pdf,
    split_documents,
    create_vectorstore,
    retrieve_documents,
    generate_answer,
)

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- Session State ---------------- #

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.title("📄 PDF Assistant")
    st.caption("Upload a PDF and chat with it using AI.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("📄 Process Document", use_container_width=True):

            with st.spinner("Processing document..."):

                documents = load_pdf(uploaded_file)
                chunks = split_documents(documents)
                vectorstore = create_vectorstore(chunks)

                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []

            st.success("✅ Document processed successfully!")

    st.divider()

    if st.session_state.documents:

        st.subheader("📑 Document Info")

        st.markdown(
            f"""
📄 **Pages:** {len(st.session_state.documents)}

✂ **Chunks:** {len(st.session_state.chunks)}

🟢 **Status:** Ready
"""
        )

        st.divider()

        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("📄 New Document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.documents = None
            st.session_state.chunks = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.caption("🚀 Powered by")
    st.caption("LangChain • FAISS")
    st.caption("HuggingFace • Groq") 


# ---------------- Main ---------------- #

st.title("📄 RAG Document Question Answering")

st.markdown(
    """
Ask questions from your uploaded PDF using **Retrieval-Augmented Generation (RAG)**.

Upload once • Process once • Ask unlimited questions 🚀
"""
)

if st.session_state.vectorstore is None:

    st.info("👈 Upload a PDF and click **Process PDF** to begin.")

    st.stop()

# Display previous chat

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            with st.expander("📚 View Sources"):

                for i, doc in enumerate(message["sources"]):

                    st.markdown(f"**Chunk {i+1}**")

                    st.write(doc.page_content)

# Chat input

question = st.chat_input("Ask anything about your PDF...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.spinner("🔍 Searching document and generating answer..."):

        retrieved_docs = retrieve_documents(
            st.session_state.vectorstore,
            question
        )

        answer = generate_answer(
            question,
            retrieved_docs
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_docs
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("📚 View Sources"):

            for i, doc in enumerate(retrieved_docs):

                st.markdown(f"**Chunk {i+1}**")

                st.write(doc.page_content)

st.divider()

st.caption(
    "Built with ❤️ using Streamlit • LangChain • FAISS • Groq • Llama 3.1"
)