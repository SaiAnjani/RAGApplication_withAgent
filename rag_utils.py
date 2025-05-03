from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

VECTOR_STORE_PATH = "vector_store"
EMBEDDINGS_PATH = "embeddings.pkl"

class SentenceTransformerEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def __call__(self, text):
        """Make the class callable for FAISS compatibility."""
        return self.embed_query(text)
        
    def embed_query(self, text):
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_documents(pdf_paths, save_to_disk=True):
    """Process PDF documents and create a vector store."""
    # Extract text from all PDFs
    texts = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        texts.append(text)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents(texts)
    
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings()
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    if save_to_disk:
        # Save vector store
        vectorstore.save_local(VECTOR_STORE_PATH)
        # Save only the model name
        model_info = {
            "model_name": "all-MiniLM-L6-v2"
        }
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(model_info, f)
    
    return vectorstore

def load_vector_store():
    """Load vector store from disk if it exists."""
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(EMBEDDINGS_PATH):
        try:
            # Load model info
            with open(EMBEDDINGS_PATH, 'rb') as f:
                model_info = pickle.load(f)
            
            # Create new embeddings instance
            embeddings = SentenceTransformerEmbeddings()
            
            # Load vector store with safe deserialization
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings,
                allow_dangerous_deserialization=True  # Safe since we created the files
            )
            return vectorstore
        except (AttributeError, pickle.UnpicklingError):
            # If there's an error loading the old format, delete the files and return None
            print("Old format detected. Deleting existing vector store and embeddings.")
            import shutil
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            if os.path.exists(EMBEDDINGS_PATH):
                os.remove(EMBEDDINGS_PATH)
            return None
    return None

def get_answer(question, vectorstore):
    """Get answer to a question using the vector store."""
    # Get relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    
    # Combine the relevant documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create a simple answer based on the most relevant context
    answer = f"Based on the invoice content:\n\n{context}"
    
    return answer 