import os
import streamlit as st
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

# Configurations
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
FAISS_INDEX_PATH = 'faiss_index.faiss'
BOOKS_DIRECTORY = 'Books'
DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Prompt Template for RAG
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant for the subject Microcontroller & Applications in B.Tech Electronics and Telecommunication Engineering.  

Given the following context from documents:  
{context}  

Answer the query strictly based on the context provided. If the context does not contain enough information, acknowledge this and provide a general response. If you use information outside the context from your own knowledge, explicitly mention it at the start of your response in parentheses.  

**Syllabus Overview:**  
1. **ARM7 Architecture:** Features of ARM core architecture, Data Flow Model, Pipeline, Registers, Operating Modes.  
2. **Introduction to ARM Programming:** THUMB introduction, ARM vs. THUMB differences, Register usage in THUMB, ARM Thumb interworking. Assembly structure, directives, and simple ALPs (arithmetic/logical operations, factorial, string operations, sorting, searching, scanning).  
3. **LPC2148 ARM CPU:** Salient features, Pin diagram, block diagram, memory mapping. Functional features: Interrupt controller, RTC, USB, UART, I2C, SPI, SSP controllers, watchdog timers, and system control unit.  
4. **LPC2148 Peripherals:** Registers, GPIOs, PLL features and structure, Timers: architecture and applications. Simple C programs for GPIO, PLL, Timer.  
5. **LPC2148 Applications:** System design examples using GPIOs (LED blinking, stepper/DC motor control, LCD interface, ADC/DAC, UART applications).  

Provide concise, accurate answers in a structured format. Use technical terminology, and for programming queries, include error-free and well-commented code snippets in ARM Assembly or C where applicable. Avoid unnecessary details unless explicitly requested.  

Query: {query}  

Answer:"""


# Load embedding model and FAISS index
@st.cache_resource
def load_index():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    return faiss_index, embedding_model

# Create ensemble retriever
@st.cache_resource
def create_ensemble_retriever(_faiss_index, _embedding_model):
    docs = list(_faiss_index.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(docs)
    faiss_retriever = _faiss_index.as_retriever(search_kwargs={'k': 4})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.25, 0.75]
    )
    
    return ensemble_retriever

# Retrieve relevant chunks
def retrieve_relevant_chunks(query: str, ensemble_retriever, top_k: int = 4) -> str:
    results = ensemble_retriever.get_relevant_documents(query)[:top_k]
    return "\n\n".join([result.page_content for result in results])

# Index new documents
def index_new_documents(uploaded_files):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    try:
        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    except:
        faiss_index = None
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(BOOKS_DIRECTORY, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        all_text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                all_text += page.get_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_text(all_text)
        
        if faiss_index is None:
            faiss_index = FAISS.from_texts(chunks, embedding_model)
        else:
            temp_index = FAISS.from_texts(chunks, embedding_model)
            faiss_index.merge_from(temp_index)
    
    faiss_index.save_local(FAISS_INDEX_PATH)
    return faiss_index

# Chat with Mixtral
def chat_with_mixtral(client, messages):
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL, 
            messages=messages, 
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
def main():
    st.sidebar.title("Your MCA Assistant")
    
    # Mixtral API Key Input
    api_key = st.sidebar.text_input("Hugging Face API Key", type="password")
    
    # Document Upload
    st.sidebar.subheader("Upload New Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner('Indexing documents...'):
            try:
                faiss_index = index_new_documents(uploaded_files)
                st.sidebar.success(f"Successfully indexed {len(uploaded_files)} documents!")
            except Exception as e:
                st.sidebar.error(f"Error indexing documents: {e}")

    # Books
    books = [f for f in os.listdir(BOOKS_DIRECTORY) if f.endswith('.pdf')]
    st.sidebar.subheader("Available Books")
    for book in books:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.sidebar.write(book)
        with col2:
            with open(os.path.join(BOOKS_DIRECTORY, book), 'rb') as f:
                st.sidebar.download_button(
                    label="Download",
                    data=f.read(),
                    file_name=book,
                    mime='application/pdf'
                )
    
    
    # Chat Interface
    st.title("MCA Conversational Assistant")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if api_key:
        try:
            client = InferenceClient(token=api_key)

            # RAG Setup
            faiss_index, embedding_model = load_index()
            ensemble_retriever = create_ensemble_retriever(faiss_index, embedding_model)

            # User input
            if prompt := st.chat_input("Ask me anything about your documents"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Retrieve context
                context = retrieve_relevant_chunks(prompt, ensemble_retriever)

                # Prepare messages for Mixtral
                messages = [
                    {
                    "role": "user",
                    "content": RAG_PROMPT_TEMPLATE.format(context=context, query=prompt)
                    }
                ]

            # Generate response using chat_with_mixtral
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_mixtral(client, messages)
                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error initializing Mixtral: {e}")
    else:
        st.info("Please enter your Hugging Face API key in the sidebar.")

if __name__ == "__main__":
    main()
