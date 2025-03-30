import os
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Streamlit UI for user input (Pinecone API key, OpenAI API key, and index name)
st.title("Query Pinecone Index")

pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
index_name = st.text_input("Enter your Pinecone Index Name", value="quickstar2")

# If keys and index name are provided
if pinecone_api_key and openai_api_key and index_name:
    # Initialize Pinecone with the API key provided by the user
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize the index
    index = pc.Index(index_name)

    # Initialize the embeddings model and Pinecone vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Streamlit UI to accept query
    query = st.text_input("Ask a question:")

    if query:
        # Set up retriever for similarity-based search
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",  # Similarity-based search
            search_kwargs={"k": 5, "score_threshold": 0.6},  # k=5 means top 5 results, score_threshold is for relevance
        )

        # Perform the search (retrieve relevant chunks)
        results = retriever.invoke(query)

        # Display the results
        if results:
            st.write("Results:")
            for res in results:
                st.write(f"**Content**: {res.page_content}")
                st.write(f"**Metadata**: {res.metadata}")
                st.write(f"**Score**: {res.score}")
        else:
            st.write("No relevant results found.")
else:
    st.warning("Please enter your Pinecone API key, OpenAI API key, and Pinecone index name.")

