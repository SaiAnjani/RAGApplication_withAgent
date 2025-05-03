import streamlit as st
import os
from PyPDF2 import PdfReader
from rag_utils import process_documents, get_answer, load_vector_store
import tempfile

st.set_page_config(page_title="Invoice Q&A", page_icon="📄", layout="wide")

st.title("📄 Invoice Q&A System")
st.write("Upload your invoices and ask questions about them!")

# Initialize session state for processed documents
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = None

# Try to load existing vector store
if st.session_state.processed_docs is None:
    st.session_state.processed_docs = load_vector_store()
    if st.session_state.processed_docs is not None:
        st.success("Loaded existing knowledge base!")

# Create tabs for different modes
tab1, tab2 = st.tabs(["Query Mode", "Processing Mode"])

with tab1:
    st.header("Query Mode")
    if st.session_state.processed_docs is not None:
        question = st.text_input("Ask a question about your invoices:")
        
        if question:
            with st.spinner("Searching for answers..."):
                answer = get_answer(question, st.session_state.processed_docs)
                st.write("Answer:", answer)
    else:
        st.info("No knowledge base found. Please process some documents in Processing Mode first.")

with tab2:
    st.header("Processing Mode")
    st.write("Upload new documents to add to the knowledge base")
    
    uploaded_files = st.file_uploader("Upload your invoices (PDF files)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_file.write(uploaded_file.getvalue())
                    temp_files.append(temp_file.name)
                
                # Process documents
                st.session_state.processed_docs = process_documents(temp_files, save_to_disk=True)
                
                # Clean up temporary files
                for temp_file in temp_files:
                    os.unlink(temp_file)
                
                st.success("Documents processed and added to knowledge base!") 