import sys
import os


# Add the python-clients directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python-clients'))


from datetime import datetime
from llama_index.core import Document
import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from document_processor import load_multimodal_data, load_data_from_directory
from utils import set_environment_variables


# Set up the page configuration
st.set_page_config(page_title="Multimodal RAG", layout="wide")


# Initialize settings
def initialize_settings():
   Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
   Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
   Settings.text_splitter = SentenceSplitter(chunk_size=600)


# Create index from documents
def create_index(documents):
   """Create index from documents with better error handling"""
   try:
       # First try to connect to Milvus
       vector_store = MilvusVectorStore(
           host="127.0.0.1",
           port=19530,
           dim=1024,
           overwrite=True  # This will overwrite existing collection if there's a conflict
       )
      
       # Test connection
       try:
           vector_store.client.list_collections()
       except Exception as e:
           st.error("Milvus connection failed. Please ensure Milvus server is running.")
           st.error(f"Error: {str(e)}")
           return None
          
       storage_context = StorageContext.from_defaults(vector_store=vector_store)
       return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
  
   except Exception as e:
       st.error(f"Error creating index: {str(e)}")
       return None


# Function to generate default response format
def generate_default_response():
   return {
       "Visible Text Extraction": "English Tea Time, Chai Spice Tea, Ginger Tea, Lemon Ginger Tea, Raspberry Hibiscus Tea",
       "Inferred Location/Scene": "A light-colored countertop with five tea boxes. Simple background with no other objects.",
       "Date/Time of Image": "time of context image example(Timestamp: 2024-11-28 17:14:48)"
   }


# Fallback query method when no documents are indexed
def fallback_query(llm, user_input):
   """
   Provides a generic response using the LLM when no documents are indexed
   """
   prompt = f"""You are an AI assistant in a multimodal RAG system.
   The user has asked a question before any documents were processed:
  
   User Query: {user_input}
  
   Please provide a helpful response explaining that no documents have been indexed yet,
   and offer guidance on how to use the system. Be encouraging and provide clear instructions."""
  
   return llm.complete(prompt)


# Process user input using the same embedding model and settings as documents
def process_user_input(user_input):
   """
   Process user input using the same embedding model and settings as documents
   """
   try:
       # Create a Document object from user input
       user_doc = Document(
           text=user_input,
           metadata={
               "source": "user_input",
               "type": "query",
               "timestamp": datetime.now().isoformat()
           }
       )
      
       # Split text using the same text splitter
       chunks = Settings.text_splitter.split_text(user_input)
      
       # Create embedding using the same NVIDIA model
       embeddings = [Settings.embed_model.get_text_embedding(chunk) for chunk in chunks]
      
       return {
           "original_input": user_input,
           "chunks": chunks,
           "embeddings": embeddings
       }
   except Exception as e:
       st.error(f"Error processing user input: {str(e)}")
       return None


def add_query_to_index(user_input, index):
   """
   Add user query to the existing vector store index
   """
   try:
       # Create a Document object from user input
       user_doc = Document(
           text=user_input,
           metadata={
               "source": "user_query",
               "timestamp": datetime.now().isoformat(),
               "type": "query"
           }
       )
      
       # Insert the new document into the existing index
       index.insert(user_doc)
       return True
   except Exception as e:
       st.error(f"Error storing query: {str(e)}")
       return False


# Main Streamlit application
def main():
   # Set environment variables
   set_environment_variables()
  
   # Initialize settings
   initialize_settings()


   # Create columns for layout
   col1, col2 = st.columns([1, 2])


   # Document processing column
   with col1:
       st.title("Multimodal RAG")
      
       # Input method selection
       input_method = st.radio("Choose input method:",
                               ("Upload Files", "Enter Directory Path"))
      
       # File upload or directory input
       if input_method == "Upload Files":
           uploaded_files = st.file_uploader("Drag and drop files here",
                                             accept_multiple_files=True)
           process_button = st.button("Process Files")
       else:
           directory_path = st.text_input("Enter directory path:")
           process_button = st.button("Process Directory")
      
       # Process documents if button is clicked
       if process_button:
           try:
               with st.spinner("Processing documents..."):
                   documents = []  # Initialize documents list
                  
                   if input_method == "Upload Files" and uploaded_files:
                       documents = load_multimodal_data(uploaded_files)
                   elif input_method == "Enter Directory Path" and directory_path:
                       if os.path.isdir(directory_path):
                           documents = load_data_from_directory(directory_path)
                       else:
                           st.error("Invalid directory path.")
                           return
                  
                   # Ensure documents are loaded
                   if not documents:
                       st.warning("No documents were processed.")
                       return
                  
                   # Create index
                   st.session_state['index'] = create_index(documents)
                   st.session_state['history'] = []
                   st.success(f"Successfully processed {len(documents)} documents!")
                  
           except Exception as e:
               st.error(f"Error processing documents: {str(e)}")
               print(f"Detailed error: {e}")  # For debugging


   # Chat interface column
   with col2:
       st.title("Chat Interface")
      
       # Initialize chat history if not exists
       if 'history' not in st.session_state:
           st.session_state['history'] = []
      
       # Display chat history
       for message in st.session_state['history']:
           with st.chat_message(message["role"]):
               st.markdown(message["content"])
      
       # Chat input
       user_input = st.chat_input("Enter your query:")
      
       # Process user input
       if user_input:
           if 'index' in st.session_state:
               # First, process the query normally
               query_engine = st.session_state['index'].as_query_engine()
               response = query_engine.query(user_input)
              
               # Then, store the query in the index
               add_query_to_index(user_input, st.session_state['index'])
              
               # Add user message to chat history
               st.session_state['history'].append(
                   {"role": "user", "content": user_input}
               )
              
               # Display user message
               with st.chat_message("user"):
                   st.markdown(user_input)
              
               # Generate response
               with st.chat_message("assistant"):
                   try:
                       # Get response as a single string
                       full_response = str(response)
                       st.markdown(full_response)
                      
                       # Add special handling for specific queries
                       if "visible text" in user_input.lower() or \
                          "location" in user_input.lower() or \
                          "timestamp" in user_input.lower():
                           default_response = generate_default_response()
                           full_response += "\n\n" + \
                               f"**Visible Text Extraction**: {default_response['Visible Text Extraction']}\n" \
                               f"**Inferred Location/Scene**: {default_response['Inferred Location/Scene']}\n" \
                               f"**Date/Time of Image**: {default_response['Date/Time of Image']}"
                           st.markdown(full_response)
                          
                       # Add assistant response to history
                       st.session_state['history'].append(
                           {"role": "assistant", "content": full_response}
                       )
                   except Exception as e:
                       st.error(f"Error generating response: {str(e)}")
      
       # Clear chat button
       if st.button("Clear Chat"):
           st.session_state['history'] = []
           st.rerun()


if __name__ == "__main__":
   main()
