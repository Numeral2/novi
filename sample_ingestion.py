import os
import time
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO

# Streamlit UI for user input (Pinecone API key and index name)
st.title("Document Ingestion for Pinecone")

# User inputs for API keys and index name
pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
index_name = st.text_input("Enter your Pinecone Index Name", value="quickstar2")

# File upload section for PDFs
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

# If all keys and index name are provided and files are uploaded
if pinecone_api_key and openai_api_key and index_name and uploaded_files:
    # Initialize Pinecone with the API key provided by the user
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists, and create it if not
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        st.write(f"Creating Pinecone index: {index_name}...")
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

    # Load and process the uploaded PDF files
    document_count = 0  # Count to keep track of the number of documents processed
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        pdf_file = BytesIO(file_content)

        # Use PyPDFDirectoryLoader to load the uploaded PDF from the BytesIO object
        loader = PyPDFDirectoryLoader(pdf_file)
        raw_documents = loader.load()

        # Split the documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Adjust based on your document size and requirements
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )

        # Split documents into chunks
        chunks = text_splitter.split_documents(raw_documents)

        # Generate unique IDs for each chunk and add them to Pinecone
        uuids = [f"id_{document_count + i}" for i in range(len(chunks))]

        # Add chunks to Pinecone with metadata (e.g., document source, chunk index)
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": uploaded_file.name,
                "chunk_index": i,
                "page_number": i + 1,  # Optional: You can add more metadata like page numbers
            }
            document = Document(page_content=chunk, metadata=metadata)
            vector_store.add_documents(documents=[document], ids=[uuids[i]])

        document_count += len(chunks)

    st.success(f"Successfully added {document_count} chunks to Pinecone.")
else:
    st.warning("Please enter your Pinecone API key, OpenAI API key, Pinecone index name, and upload at least one PDF file.")

