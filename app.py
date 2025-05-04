import streamlit as st
import os
from PyPDF2 import PdfReader
from rag_utils import process_documents, get_answer, load_vector_store
import tempfile

st.set_page_config(page_title="Invoice Q&A", page_icon="��", layout="wide")

# Initialize session state for processed documents and entities
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = None
if 'entities' not in st.session_state:
    st.session_state.entities = None

# Try to load existing vector store and entities
if st.session_state.processed_docs is None:
    result = load_vector_store()
    if result is not None:
        st.session_state.processed_docs, st.session_state.entities = result
        st.success("Loaded existing knowledge base!")
    else:
        st.info("No existing knowledge base found. Please process some documents in Processing Mode first.")

# Create tabs for different modes
tab1, tab2 = st.tabs(["📝 Process Documents", "🔍 View Entities"])

with tab1:
    st.title("📄 Invoice Q&A System")
    st.write("Upload your invoices and ask questions about them!")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF invoices", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        # Create temporary files
        temp_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)

        # Process documents button
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                st.session_state.processed_docs, st.session_state.entities = process_documents(temp_files, save_to_disk=True)
                st.success("Documents processed successfully!")
            
            # Clean up temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)

    # Load existing knowledge base
    if st.session_state.processed_docs is None:
        result = load_vector_store()
        if result is not None:
            st.session_state.processed_docs, st.session_state.entities = result
            st.info("Loaded existing knowledge base.")
        else:
            st.info("No existing knowledge base found. Please process some documents first.")

    # Question input
    if st.session_state.processed_docs is not None:
        st.subheader("Ask a Question")
        question = st.text_input("Enter your question about the invoices:")
        
        if question:
            with st.spinner("Generating answer..."):
                answer = get_answer(question, st.session_state.processed_docs, st.session_state.entities)
                st.write(answer)

with tab2:
    st.title("🔍 Extracted Entities")
    
    if st.session_state.entities is not None:
        # Create a container for the entities table
        container = st.container()
        
        # Display entities for each document
        for doc_name, entities in st.session_state.entities.items():
            with container:
                st.subheader(f"Document: {doc_name}")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Basic Information
                    st.write("**Basic Information**")
                    info_data = {
                        "Invoice Number": entities.get('invoice_number', 'Not found'),
                        "Date": entities.get('date', 'Not found'),
                        "Total Amount": entities.get('total_amount', 'Not found')
                    }
                    st.table(info_data)
                
                with col2:
                    # Vendor Information
                    st.write("**Vendor Information**")
                    vendor_data = {
                        "Vendor Name": entities.get('vendor_name', 'Not found'),
                        "Vendor Address": "[ADDRESS]" if entities.get('vendor_address') else 'Not found'
                    }
                    st.table(vendor_data)
                
                # Customer Information
                st.write("**Customer Information**")
                customer_data = {
                    "Customer Name": entities.get('customer_name', 'Not found'),
                    "Customer Address": "[ADDRESS]" if entities.get('customer_address') else 'Not found'
                }
                st.table(customer_data)
                
                # Line Items
                if entities.get('items'):
                    st.write("**Line Items**")
                    items_data = []
                    for item in entities['items']:
                        items_data.append({
                            "Description": item.get('description', 'Not found'),
                            "Amount": item.get('amount', 'Not found')
                        })
                    st.table(items_data)
                
                st.divider()
    else:
        st.info("No entities found. Please process some documents first.") 