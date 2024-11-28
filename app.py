import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from typing import List

# Load embedding model and FAISS index
@st.cache_resource
def load_index():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.load_local('faiss_index.faiss', embedding_model, allow_dangerous_deserialization=True)
    return faiss_index, embedding_model

# Create ensemble retriever
@st.cache_resource
def create_ensemble_retriever(faiss_index, embedding_model):
    # Convert FAISS index documents to list for BM25
    docs = faiss_index.docstore._dict.values()
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    # Create FAISS retriever
    faiss_retriever = faiss_index.as_retriever(search_kwargs={'k': 4})
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

# Retrieve relevant chunks
def retrieve_relevant_chunks(query: str, ensemble_retriever, top_k: int = 4) -> List[str]:
    results = ensemble_retriever.get_relevant_documents(query)[:top_k]
    return [result.page_content for result in results]

# Streamlit app
def main():
    st.title("Document Retrieval System")
    
    # Load index and create retrievers
    try:
        faiss_index, embedding_model = load_index()
        ensemble_retriever = create_ensemble_retriever(faiss_index, embedding_model)
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return
    
    # Search input
    query = st.text_input("Enter your search query:")
    
    # Retrieve and display results
    if query:
        try:
            # Retrieve relevant chunks
            result_chunks = retrieve_relevant_chunks(query, ensemble_retriever)
            
            # Display results
            st.subheader("Retrieved Chunks:")
            for i, chunk in enumerate(result_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
                st.markdown("---")
        
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")

if __name__ == "__main__":
    main()