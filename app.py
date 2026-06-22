import streamlit as st
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document

# Handle the import for text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        import re
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=150, length_function=len):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
            
            def split_text(self, text):
                chunks = []
                start = 0
                text_length = self.length_function(text)
                
                while start < text_length:
                    end = min(start + self.chunk_size, text_length)
                    if end < text_length:
                        paragraph_break = text.rfind('\n\n', start, end)
                        if paragraph_break != -1 and paragraph_break > start:
                            end = paragraph_break + 2
                        else:
                            newline = text.rfind('\n', start, end)
                            if newline != -1 and newline > start:
                                end = newline + 1
                    
                    chunks.append(text[start:end].strip())
                    start = end - self.chunk_overlap if end < text_length else text_length
                
                return [chunk for chunk in chunks if chunk]

# Set page configuration
st.set_page_config(
    page_title="TETFund Document Query System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 TETFund Conditions of Service Query System")
st.markdown("""
This application allows you to ask questions about the TETFund Staff Conditions of Service document.
The system uses RAG (Retrieval-Augmented Generation) to find relevant information and generate answers.
""")

@st.cache_resource
def load_and_process_document(doc_path):
    """Load the document, split into chunks, and create embeddings."""
    try:
        doc = Document(doc_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        if not text.strip():
            st.error("The document appears to be empty.")
            return None, None, None, None, None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        st.info(f"✅ Document loaded! Created {len(chunks)} chunks.")
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        chunk_embeddings = embedding_model.encode(chunks)
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        generation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        return chunks, chunk_embeddings, embedding_model, tokenizer, generation_model
        
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

def retrieve_context(question, chunks, embeddings, model, k=3):
    """Retrieve relevant chunks using cosine similarity."""
    try:
        # Encode the question
        question_embedding = model.encode([question])
        
        # Calculate cosine similarity
        from numpy.linalg import norm
        similarities = np.dot(embeddings, question_embedding.T).flatten()
        norms = norm(embeddings, axis=1) * norm(question_embedding, axis=1)
        similarities = similarities / norms
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        retrieved_chunks = [chunks[i] for i in top_indices]
        
        return "\n\n".join(retrieved_chunks)
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    uploaded_file = st.file_uploader(
        "Upload a DOCX file",
        type=['docx'],
        help="Upload the TETFund Conditions of Service document"
    )
    
    default_path = st.text_input(
        "Or enter document path:",
        value="CoS.docx",
        help="Path to the document on your system"
    )
    
    k = st.slider(
        "Number of chunks to retrieve:",
        min_value=1,
        max_value=10,
        value=3
    )
    
    st.divider()
    
    if st.button("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared! Refresh the page.")
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "💬 Ask a question:",
        placeholder="e.g., What is the policy on promotion?",
        height=100
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the document and generating answer..."):
                try:
                    embedding_model, tokenizer, generation_model = load_models()
                    if embedding_model is None:
                        st.error("Failed to load models.")
                        st.stop()
                    
                    doc_path = None
                    if uploaded_file is not None:
                        temp_path = "temp_document.docx"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        doc_path = temp_path
                    else:
                        doc_path = default_path
                    
                    chunks, embeddings, _, _, _ = load_and_process_document(doc_path)
                    
                    if chunks is None:
                        st.error("Failed to load document.")
                        st.stop()
                    
                    context = retrieve_context(question, chunks, embeddings, embedding_model, k)
                    
                    if not context:
                        st.warning("No relevant context found.")
                        st.stop()
                    
                    prompt = f"""
                    Answer the question using only the context below.
                    Do not guess. If the answer is not clearly stated, say you do not know.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Answer:
                    """
                    
                    answer = generate_answer(prompt, tokenizer, generation_model)
                    
                    st.subheader("📝 Answer")
                    st.success(answer)
                    
                    with st.expander("📚 View Retrieved Context"):
                        st.text_area("Context used:", context, height=300)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with col2:
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is the policy on promotion?",
        "What are the eligibility requirements for appointment?",
        "What types of leave are available?",
        "What is the disciplinary procedure?",
        "What is the policy on staff training?",
        "How are employees compensated?",
        "What is the retirement age?",
        "What loans are available to employees?"
    ]
    
    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state.question = eq
            st.rerun()

if 'question' not in st.session_state:
    st.session_state.question = ""

st.divider()
st.markdown("""
**Note:** Uses RAG with:
- **Embedding Model:** all-MiniLM-L6-v2
- **Generation Model:** FLAN-T5-small
- **Search:** Cosine similarity
""")
