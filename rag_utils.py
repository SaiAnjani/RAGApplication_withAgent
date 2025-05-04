from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import re
from mcp_framework import MCPFramework
import time

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

VECTOR_STORE_PATH = "vector_store"
EMBEDDINGS_PATH = "embeddings.pkl"
ENTITIES_PATH = "entities.pkl"

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

def extract_entities(text):
    """Extract key entities from invoice text."""
    entities = {
        'invoice_number': None,
        'date': None,
        'total_amount': None,
        'vendor_name': None,
        'vendor_address': None,
        'customer_name': None,
        'customer_address': None,
        'items': []
    }
    
    # Extract invoice number (common patterns)
    invoice_patterns = [
        r'Invoice\s*#?\s*[:#]?\s*([A-Z0-9-]+)',
        r'INV\s*#?\s*[:#]?\s*([A-Z0-9-]+)',
        r'Invoice\s*Number\s*[:#]?\s*([A-Z0-9-]+)'
    ]
    for pattern in invoice_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['invoice_number'] = match.group(1).strip()
            break
    
    # Extract date (common formats)
    date_patterns = [
        r'Date\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'Invoice\s*Date\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['date'] = match.group(1).strip()
            break
    
    # Extract total amount
    total_patterns = [
        r'Total\s*Amount\s*[:#]?\s*\$?\s*([\d,]+\.\d{2})',
        r'Amount\s*Due\s*[:#]?\s*\$?\s*([\d,]+\.\d{2})',
        r'Total\s*[:#]?\s*\$?\s*([\d,]+\.\d{2})'
    ]
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities['total_amount'] = match.group(1).strip()
            break
    
    # Extract vendor information
    vendor_section = re.search(r'From:?\s*(.*?)(?=To:|$)', text, re.DOTALL | re.IGNORECASE)
    if vendor_section:
        vendor_text = vendor_section.group(1)
        # Extract vendor name (first line)
        vendor_name = vendor_text.split('\n')[0].strip()
        entities['vendor_name'] = vendor_name
        # Extract vendor address (remaining lines)
        vendor_address = '\n'.join(vendor_text.split('\n')[1:]).strip()
        entities['vendor_address'] = vendor_address
    
    # Extract customer information
    customer_section = re.search(r'To:?\s*(.*?)(?=From:|$)', text, re.DOTALL | re.IGNORECASE)
    if customer_section:
        customer_text = customer_section.group(1)
        # Extract customer name (first line)
        customer_name = customer_text.split('\n')[0].strip()
        entities['customer_name'] = customer_name
        # Extract customer address (remaining lines)
        customer_address = '\n'.join(customer_text.split('\n')[1:]).strip()
        entities['customer_address'] = customer_address
    
    # Extract line items (basic pattern)
    items_section = re.search(r'Description.*?Amount', text, re.DOTALL | re.IGNORECASE)
    if items_section:
        items_text = items_section.group(0)
        # Split into lines and process each line
        lines = items_text.split('\n')
        for line in lines:
            # Look for lines with numbers (likely containing amounts)
            if re.search(r'\d+\.\d{2}', line):
                # Extract description and amount
                parts = line.split()
                if len(parts) >= 2:
                    amount = parts[-1]
                    description = ' '.join(parts[:-1])
                    entities['items'].append({
                        'description': description,
                        'amount': amount
                    })
    
    return entities

def process_documents(pdf_paths, save_to_disk=True):
    """Process PDF documents and create a vector store."""
    # Extract text from all PDFs
    texts = []
    sources = []
    entities = {}  # Dictionary to store entities for each document
    
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        texts.append(text)
        source = os.path.basename(pdf_path)
        sources.append(source)
        
        # Extract entities from the document
        doc_entities = extract_entities(text)
        entities[source] = doc_entities
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for i, text in enumerate(texts):
        doc_chunks = text_splitter.create_documents([text])
        # Add source metadata to each chunk
        for chunk in doc_chunks:
            chunk.metadata = {"source": sources[i]}
        chunks.extend(doc_chunks)
    
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
        # Save entities
        with open(ENTITIES_PATH, 'wb') as f:
            pickle.dump(entities, f)
    
    return vectorstore, entities

def load_vector_store():
    """Load vector store and entities from disk if they exist."""
    # Check if all required files exist
    if not all(os.path.exists(path) for path in [VECTOR_STORE_PATH, EMBEDDINGS_PATH, ENTITIES_PATH]):
        print("Required files not found. Starting fresh.")
        return None
    
    try:
        # Load model info
        with open(EMBEDDINGS_PATH, 'rb') as f:
            model_info = pickle.load(f)
        
        # Load entities
        with open(ENTITIES_PATH, 'rb') as f:
            entities = pickle.load(f)
        
        # Create new embeddings instance
        embeddings = SentenceTransformerEmbeddings()
        
        # Load vector store with safe deserialization
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore, entities
    except (AttributeError, pickle.UnpicklingError, Exception) as e:
        print(f"Error loading files: {e}")
        # Clean up corrupted files
        import shutil
        for path in [VECTOR_STORE_PATH, EMBEDDINGS_PATH, ENTITIES_PATH]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        return None

def get_answer(question, vectorstore, entities):
    """Get answer to a question using the vector store and synthesize the response with Gemini."""
    # Get relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    
    # Extract relevant information from documents
    relevant_info = []
    for doc in docs:
        # Split the content into sentences for better processing
        sentences = doc.page_content.split('. ')
        # Keep only sentences that seem relevant to the question
        relevant_sentences = [s for s in sentences if any(word.lower() in s.lower() for word in question.split())]
        if relevant_sentences:
            source = doc.metadata.get('source', 'Unknown')
            relevant_info.append({
                'content': '. '.join(relevant_sentences),
                'source': source,
                'entities': entities.get(source, {})
            })
    
    # Synthesize the answer
    if not relevant_info:
        return "I couldn't find any relevant information in the documents to answer your question."
    
    # Define address patterns for both try and except blocks
    address_patterns = [
        r'\d+[,\s]+(?:[A-Za-z\s]+(?:Wing|Floor|Gate|Block|Building|Tower|Phase|Sector|Area|Colony|Layout|Estate|Acres|Gardens|Heights|Residency|Residence|Residential|Complex|Apartment|Flat|Suite|Unit|Room|Floor|Level|Storey|St|Ave|Avenue|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Wy|Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Wy)[,\s]+)*[A-Za-z\s]+',
        r'Address:.*?(?=\n|$)',
        r'Location:.*?(?=\n|$)',
        r'Building:.*?(?=\n|$)',
        r'Floor:.*?(?=\n|$)',
        r'Wing:.*?(?=\n|$)',
        r'Gate:.*?(?=\n|$)',
        r'Block:.*?(?=\n|$)',
        r'Phase:.*?(?=\n|$)',
        r'Sector:.*?(?=\n|$)',
        r'Area:.*?(?=\n|$)',
        r'Colony:.*?(?=\n|$)',
        r'Layout:.*?(?=\n|$)',
        r'Estate:.*?(?=\n|$)',
        r'Acres:.*?(?=\n|$)',
        r'Gardens:.*?(?=\n|$)',
        r'Heights:.*?(?=\n|$)',
        r'Residency:.*?(?=\n|$)',
        r'Residence:.*?(?=\n|$)',
        r'Residential:.*?(?=\n|$)',
        r'Complex:.*?(?=\n|$)',
        r'Apartment:.*?(?=\n|$)',
        r'Flat:.*?(?=\n|$)',
        r'Suite:.*?(?=\n|$)',
        r'Unit:.*?(?=\n|$)',
        r'Room:.*?(?=\n|$)',
        r'Floor:.*?(?=\n|$)',
        r'Level:.*?(?=\n|$)',
        r'Storey:.*?(?=\n|$)'
    ]
    
    # Prepare context for Gemini
    context = "\n\n".join([f"Source: {info['source']}\nContent: {info['content']}\nEntities: {info['entities']}" for info in relevant_info])
    
    # Create prompt for Gemini
    prompt = f"""You are an expert at synthesizing information from documents. Based on the following context from invoice documents, provide a clear and concise answer to the question: '{question}'

Context:
{context}

IMPORTANT: The system has already extracted structured entities from the documents. These entities contain precise information about:
- Invoice details (number, date, total amount)
- Vendor information (name, address)
- Customer information (name, address)
- Line items (descriptions and amounts)

Answer Generation Strategy:
1. PRIMARY SOURCE: Check if the question can be answered using ONLY the extracted entities
   - If yes, use the entity information directly
   - This is the most accurate and reliable source
   - Example: For "What is the Total Amount?", use the 'Total Amount' entity

2. SECONDARY SOURCE: If entities don't fully answer the question:
   - Combine relevant entity information with document content
   - Use entities as the primary source and supplement with document context
   - Example: For "What was purchased?", use 'items' entity and relevant document sections

3. FALLBACK: Only use document content if:
   - No relevant entities exist
   - Question requires broader context
   - Example: For "What is the payment terms?", use document content

Response Requirements:
1. Provide ONLY a one-line direct answer to the question
2. Remove all confidential information including:
   - Addresses (any text containing building numbers, floors, wings, gates, or location names)
   - Phone numbers
   - Email addresses
   - Personal identifiers
3. After the answer, list ONLY the relevant source documents that support your answer
4. Do not include any additional explanations or details
5. Keep the response extremely concise

Format your response exactly as follows:
[One-line answer] (Source: [document1.pdf, document2.pdf])"""

    # Get response from Gemini with retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Configure generation parameters for faster response
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
                "candidate_count": 1
            }
            
            # Generate response without timeout parameter
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            initial_answer = response.text
            print(f'initial_answer: {initial_answer}')
            
            # Process through MCP framework
            mcp = MCPFramework()
            processed_response = mcp.process_response(question, initial_answer, docs)
            
            # Format the final response
            final_response = []
            
            # Clean the CRISP answer of any remaining addresses
            crisp_answer = processed_response['crisp_answer']
            for pattern in address_patterns:
                crisp_answer = re.sub(pattern, '[ADDRESS]', crisp_answer, flags=re.IGNORECASE)
            
            final_response.append(f"CRISP Answer: {crisp_answer}")
            
            # Add supporting chunks
            if processed_response['supporting_chunks']:
                final_response.append("\nSupporting Information:")
                for i, chunk in enumerate(processed_response['supporting_chunks'], 1):
                    # Clean the chunk content of addresses
                    chunk_content = chunk['content']
                    for pattern in address_patterns:
                        chunk_content = re.sub(pattern, '[ADDRESS]', chunk_content, flags=re.IGNORECASE)
                    
                    final_response.append(f"\n{i}. {chunk_content}")
                    if chunk['source'] != 'Unknown':
                        final_response.append(f"   (Source: {chunk['source']})")
            
            return '\n'.join(final_response)
            
        except Exception as e:
            print(f'Attempt {attempt + 1} failed with error: {e}')
            if attempt < max_retries - 1:
                print(f'Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)
            else:
                print('Max retries reached. Falling back to basic response.')
                # Fallback to basic response if all retries fail
                response = []
                response.append(f"Based on the invoice content, here's what I found regarding your question: '{question}'")
                response.append("\nRelevant Information:")
                
                for i, info in enumerate(relevant_info, 1):
                    # Basic confidential information removal in fallback
                    content = info['content']
                    # Remove email addresses
                    content = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', content)
                    # Remove phone numbers
                    content = re.sub(r'(\+\d{1,3}[-.]?)?\d{3}[-.]?\d{3}[-.]?\d{4}', '[PHONE]', content)
                    # Remove addresses with enhanced patterns
                    for pattern in address_patterns:
                        content = re.sub(pattern, '[ADDRESS]', content, flags=re.IGNORECASE)
                    
                    response.append(f"\n{i}. {content}")
                    if info['source'] != 'Unknown':
                        response.append(f"   (Source: {info['source']})")
                
                return '\n'.join(response) 