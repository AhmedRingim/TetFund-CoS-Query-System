import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document

# Handle the import for text splitter - works with both old and new versions
try:
    # Try the new import first (langchain >= 0.0.200)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # Try the old import pattern
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback - implement a simple splitter ourselves
        import re
        
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=150, length_function=len):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
            
            def split_text(self, text):
                """Simple fallback splitter if langchain isn't available"""
                chunks = []
                start = 0
                text_length = self.length_function(text)
                
                while start < text_length:
                    end = min(start + self.chunk_size, text_length)
                    # Try to split at paragraph or newline
                    if end < text_length:
                        # Look for paragraph break
                        paragraph_break = text.rfind('\n\n', start, end)
                        if paragraph_break != -1 and paragraph_break > start:
                            end = paragraph_break + 2
                        else:
                            # Look for newline
                            newline = text.rfind('\n', start, end)
                            if newline != -1 and newline > start:
                                end = newline + 1
                    
                    chunks.append(text[start:end].strip())
                    start = end - self.chunk_overlap if end < text_length else text_length
                
                return chunks

# Set page configuration
st.set_page_config(
    page_title="TETFund Document Query System",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 TETFund Conditions of Service Query System")
st.markdown("""
This application allows you to ask questions about the TETFund Staff Conditions of Service document.
The system uses RAG (Retrieval-Augmented Generation) to find relevant information and generate answers.
""")

# Cache the document loading and processing
@st.cache_resource
def load_and_process_document(doc_path):
    """Load the document, split into chunks, and create embeddings."""
    try:
        # Load document
        doc = Document(doc_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        if not text.strip():
            st.error("The document appears to be empty. Please check the file.")
            return None, None, None, None, None
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        st.info(f"✅ Document loaded successfully! Created {len(chunks)} chunks.")
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings
        chunk_embeddings = embedding_model.encode(chunks)
        
        # Create FAISS index
        d = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(chunk_embeddings).astype('float32'))
        
        # Load generation model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        generation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        return chunks, index, embedding_model, tokenizer, generation_model
        
    except FileNotFoundError:
        st.error(f"File not found: {doc_path}. Please make sure the file exists.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None, None, None, None, None

@st.cache_resource
def load_models():
    """Load models separately for better caching."""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        generation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        return embedding_model, tokenizer, generation_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def generate_answer(prompt, tokenizer, generation_model, max_new_tokens=250):
    """Generate an answer using the FLAN-T5 model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = generation_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def retrieve_context(question, chunks, index, embedding_model, k=3):
    """Retrieve relevant chunks for the question."""
    try:
        question_embedding = embedding_model.encode([question])
        question_embedding = np.array(question_embedding).astype("float32")
        distances, indices = index.search(question_embedding, k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        return "\n\n".join(retrieved_chunks)
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

# Sidebar for file upload and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a DOCX file",
        type=['docx'],
        help="Upload the TETFund Conditions of Service document"
    )
    
    # Or use default path
    default_path = st.text_input(
        "Or enter document path:",
        value="CoS.docx",
        help="Path to the document on your system"
    )
    
    # Number of chunks to retrieve
    k = st.slider(
        "Number of chunks to retrieve:",
        min_value=1,
        max_value=10,
        value=3,
        help="How many document sections to use for context"
    )
    
    st.divider()
    
    if st.button("🔄 Clear Cache", help="Reload all models and documents"):
        st.cache_resource.clear()
        st.success("Cache cleared! Refresh the page to reload.")
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Question input
    question = st.text_area(
        "💬 Ask a question about the TETFund Conditions of Service:",
        placeholder="e.g., What is the policy on promotion?",
        height=100
    )
    
    # Query button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the document and generating answer..."):
                try:
                    # Load models
                    embedding_model, tokenizer, generation_model = load_models()
                    if embedding_model is None:
                        st.error("Failed to load models. Please check the logs.")
                        st.stop()
                    
                    # Determine document path
                    doc_path = None
                    if uploaded_file is not None:
                        # Save uploaded file temporarily
                        temp_path = "temp_document.docx"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        doc_path = temp_path
                    else:
                        doc_path = default_path
                    
                    # Load document
                    chunks, index, _, _, _ = load_and_process_document(doc_path)
                    
                    if chunks is None:
                        st.error("Failed to load document. Please check the file path or upload a valid file.")
                        st.stop()
                    
                    # Retrieve context
                    context = retrieve_context(question, chunks, index, embedding_model, k)
                    
                    if not context:
                        st.warning("No relevant context found for your question. Please try a different question.")
                        st.stop()
                    
                    # Create prompt
                    prompt = f"""
                    Answer the question using only the context below.
                    Do not guess. If the answer is not clearly stated, say you do not know.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Answer:
                    """
                    
                    # Generate answer
                    answer = generate_answer(prompt, tokenizer, generation_model)
                    
                    # Display results
                    st.subheader("📝 Answer")
                    st.success(answer)
                    
                    with st.expander("📚 View Retrieved Context"):
                        st.text_area("Context used for the answer:", context, height=300)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please make sure the document is accessible and in the correct format.")

with col2:
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is the policy on promotion?",
        "What are the eligibility requirements for appointment?",
        "What types of leave are available?",
        "What is the procedure for disciplinary matters?",
        "What is the policy on staff training and development?",
        "How are employees compensated?",
        "What is the retirement age?",
        "What loans are available to employees?"
    ]
    
    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state.question = eq
            st.rerun()

# Initialize session state for question
if 'question' not in st.session_state:
    st.session_state.question = ""

# Footer
st.divider()
st.markdown("""
**Note:** This system uses RAG (Retrieval-Augmented Generation) with:
- **Embedding Model:** all-MiniLM-L6-v2 for semantic search
- **Generation Model:** FLAN-T5-small for answer generation
- **Vector Search:** FAISS for efficient similarity search
""")