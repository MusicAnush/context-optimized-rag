import streamlit as st
import os
from rag_engine.py import StructuralRAGEngine
from evaluator import RAGEvaluator

st.set_page_config(page_title="Production RAG Engine Sandbox", layout="wide")

st.title("🏛️ Production-Grade Structural RAG Engine")
st.caption("A custom implementation highlighting Hybrid Retrieval, Reciprocal Rank Fusion, and Context Optimization.")

# Sidebar Configurations
with st.sidebar:
    st.header("⚙️ Configuration Engine")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.subheader("Hyperparameters")
    chunk_size = st.slider("Chunk Size (Words)", 100, 1000, 400, step=50)
    overlap = st.slider("Chunk Overlap (Words)", 10, 250, 50, step=10)
    
    st.info("💡 Framework-free pipeline: Powered completely by manual numpy manipulation, BM25Okapi, and FAISS.")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar configuration panel to test the engine.")
    st.stop()

# Initialize Engine State
if "engine" not in st.session_state:
    st.session_state.engine = None

# Step 1: Ingestion Zone
st.header("1. Document Ingestion")
sample_data = st.text_area("Paste Corpus Text", height=200, placeholder="Paste your structured documents or complex texts here...")

if st.button("Build Vector & Sparse Indices"):
    if sample_data.strip():
        with st.spinner("Processing text, computing sparse arrays, and building FAISS indices..."):
            engine = StructuralRAGEngine(openai_api_key=api_key)
            engine.ingest_document(sample_data, chunk_size=chunk_size, overlap=overlap)
            st.session_state.engine = engine
            st.success(f"Successfully processed and indexed text into {len(engine.chunks)} discrete segments!")
    else:
        st.error("Please enter valid context text.")

# Step 2: Query Execution Zone
if st.session_state.engine:
    st.markdown("---")
    st.header("2. Search & Synthesis Processing")
    query_text = st.text_input("Enter Query Expression", placeholder="Ask a question about the document structure or facts...")
    
    if query_text:
        with st.spinner("Executing retrieval streams and synthesizing response..."):
            # Execute RAG Operation
            answer, raw_chunks = st.session_state.engine.query(query_text)
            
            # Execute Evaluation Metric
            evaluator = RAGEvaluator(openai_api_key=api_key)
            joined_context = "\n".join(raw_chunks)
            faithfulness_score = evaluator.evaluate_faithfulness(query_text, joined_context, answer)
            
            # Render Layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Synthesized LLM Response")
                st.write(answer)
                
                st.metric(
                    label="System Faithfulness Metric (Groundedness Verification)", 
                    value=f"{faithfulness_score * 100:.1f}%",
                    delta="No Hallucinations Detected" if faithfulness_score > 0.85 else "Caution: Potential Hallucination Risk"
                )
            
            with col2:
                st.subheader("Optimized Context Chunks (Fused Streams)")
                for idx, chunk in enumerate(raw_chunks):
                    with st.expander(f"Retrieved Document Segment #{idx + 1}"):
                        st.write(chunk)
