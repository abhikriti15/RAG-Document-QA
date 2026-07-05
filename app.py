import streamlit as st
from rag import (
    load_pdf,
    split_documents,
    create_vectorstore,
    retrieve_documents,
    generate_answer,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="📄 RAG PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("📄 PDF Assistant")
    st.write("Upload a PDF and ask questions using AI.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("📄 Process Document", use_container_width=True):

            try:

                progress = st.progress(0)

                with st.spinner("Loading PDF..."):
                    progress.progress(20)
                    documents = load_pdf(uploaded_file)

                with st.spinner("Splitting document..."):
                    progress.progress(50)
                    chunks = split_documents(documents)

                with st.spinner("Creating vector database..."):
                    progress.progress(80)
                    vectorstore = create_vectorstore(chunks)

                progress.progress(100)

                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []

                st.success("✅ Document processed successfully!")

            except Exception as e:

                st.error("❌ Failed to process the document.")
                st.exception(e)

    st.divider()

    if st.session_state.documents:

        st.subheader("📑 Document Information")

        col1, col2 = st.columns(2)

        col1.metric("Pages", len(st.session_state.documents))
        col2.metric("Chunks", len(st.session_state.chunks))

        st.success("Ready for Questions")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):

            st.session_state.messages = []
            st.rerun()

        if st.button("📄 Upload New Document", use_container_width=True):

            st.session_state.vectorstore = None
            st.session_state.documents = None
            st.session_state.chunks = None
            st.session_state.messages = []

            st.rerun()

    st.divider()

    st.markdown("### ⚙️ Tech Stack")

    st.markdown("""
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq Llama 3.1
- Streamlit
""")

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("📄 RAG Document Question Answering")

st.markdown(
"""
Upload a PDF document and ask questions using
**Retrieval-Augmented Generation (RAG)**.

The assistant retrieves relevant document chunks before generating answers with **Groq Llama 3.1**.
"""
)

if st.session_state.vectorstore is None:

    st.info("👈 Upload a PDF and click **Process Document** to begin.")

    st.stop()

# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if "sources" in message:

                with st.expander("📚 Retrieved Source Chunks"):

                    for idx, doc in enumerate(message["sources"], start=1):

                        st.markdown(f"### Chunk {idx}")

                        st.write(doc.page_content)

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input("Ask anything about your PDF...")

if question and question.strip():

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    try:

        with st.spinner("🔍 Searching document..."):

            retrieved_docs = retrieve_documents(
                st.session_state.vectorstore,
                question
            )

        with st.spinner("🤖 Generating answer..."):

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

            st.caption(
                f"Retrieved **{len(retrieved_docs)}** relevant chunks."
            )

            with st.expander("📚 Retrieved Source Chunks"):

                for idx, doc in enumerate(retrieved_docs, start=1):

                    st.markdown(f"### Chunk {idx}")

                    st.write(doc.page_content)

    except Exception as e:

        st.error("❌ Unable to generate an answer.")

        st.exception(e)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.markdown(
"""
<div style="text-align:center">

Built with ❤️ by <b>Abhikriti Saxena</b>

<b>Streamlit</b> • <b>LangChain</b> • <b>FAISS</b> •
<b>HuggingFace</b> • <b>Groq Llama 3.1</b>

</div>
""",
unsafe_allow_html=True,
)