import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Streamlit UI for entering API keys and index name
st.title("Retrieval Augmented Generation (RAG)")

# Entering Pinecone API key, OpenAI API key, and Index name
pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
index_name = st.text_input("Enter your Pinecone Index Name")

if pinecone_api_key and openai_api_key and index_name:
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize the index
    index = pc.Index(index_name)

    # Initialize OpenAI embeddings model and Pinecone Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Set up retriever for Pinecone with similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )

    # Execute query
    query = st.text_input("Ask a question", "What is retrieval augmented generation?")
    if query:
        results = retriever.invoke(query)

        # Display the results
        st.subheader("Results:")
        for res in results:
            # Make sure the metadata is handled correctly in the results
            if 'metadata' in res:
                st.markdown(f"**Text:** {res.page_content}")
                st.markdown(f"**Metadata:** {res.metadata}")
            else:
                st.markdown(f"**Text:** {res.page_content} (No metadata available)")
else:
    st.warning("Please enter all required fields (API keys and index name).")

