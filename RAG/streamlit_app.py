import streamlit as st
import os
from dotenv import load_dotenv
from conversational_rag import (
    create_pinecone_index,
    add_documents_to_pinecone_index,
    conversational_rag_chain,
    WebBaseLoader,
    RecursiveCharacterTextSplitter,
    PineconeVectorStore,
    GoogleGenerativeAIEmbeddings
)
import bs4

# Load environment variables
load_dotenv(".env")
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# Initialize session state for storing chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def process_url(url):
    """Process a URL and add it to the vector store"""
    try:
        # Load documents from URL
        loader = WebBaseLoader(
            web_paths=(url,),
            # bs_kwargs=dict(
            #     parse_only=bs4.SoupStrainer(
            #         class_=("__next")
            #     )
            # ),
        )
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        
        # Create or get Pinecone index
        index = create_pinecone_index('webloader-index')
        vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
        
        # Add documents to vector store
        add_documents_to_pinecone_index(vector_store, split_documents)
        
        return True
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return False

# Streamlit UI
st.title("Conversational RAG Chat Interface")

# URL input section
with st.expander("Add New URL to RAG System", expanded=False):
    url = st.text_input("Enter URL to add to the RAG system:")
    if st.button("Process URL"):
        if url:
            with st.spinner("Processing URL..."):
                if process_url(url):
                    st.success("URL processed successfully!")
                else:
                    st.error("Failed to process URL")
        else:
            st.warning("Please enter a URL")

# Chat interface
st.subheader("Chat with RAG System")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "abc123"}}
            )["answer"]
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 