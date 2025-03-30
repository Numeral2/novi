import os
import streamlit as st
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Initialize Pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Set the Pinecone index
index_name = "quickstar2"  # Ensure index name matches the one used for chunking
index = pc.Index(index_name)

# Initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Streamlit UI to accept query
st.title("Query Pinecone Index")
query = st.text_input("Ask a question:")

if query:
    # Retrieval setup with similarity-based search
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",  # Ensures only relevant results are returned
        search_kwargs={"k": 5, "score_threshold": 0.6},  # k=5 means top 5 results with score threshold
    )

    # Perform the search (retrieve relevant chunks)
    results = retriever.invoke(query)

    # Display the results
    st.write("Results:")
    for res in results:
        st.write(f"**Content**: {res.page_content}")
        st.write(f"**Metadata**: {res.metadata}")
        st.write(f"**Score**: {res.score}")

