import os
import time
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Streamlit UI for user input (Pinecone API key and index name)
st.title("Document Ingestion for Pinecone")

pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
index_name = st.text_input("Enter your Pinecone Index Name", value="quickstar2")

# If all keys and index name are provided
if pinecone_api_key and openai_api_key and index_name:
    # Initialize Pinecone with the API key provided by the user
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists, and create it if not
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,  # Adjust based on your model's embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)

    # Initialize the embeddings model and Pinecone vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Load PDF or other documents using PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader("path_to_documents/")  # Path to the folder containing PDFs
    raw_documents = loader.load()

    # Split the documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Adjust based on the document size and your requirements
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(raw_documents)

    # Generate unique IDs for each chunk
    uuids = [f"id{i}" for i in range(len(chunks))]

    # Add chunks to Pinecone with metadata (e.g., document source, page number, etc.)
    for i, chunk in enumerate(chunks):
        metadata = {"source": f"Document {i}", "chunk_index": i}  # Customize metadata as needed
        document = Document(page_content=chunk, metadata=metadata)
        vector_store.add_documents(documents=[document], ids=[uuids[i]])

    st.success(f"Added {len(chunks)} chunks to Pinecone.")
else:
    st.warning("Please enter your Pinecone API key, OpenAI API key, and Pinecone index name.")

