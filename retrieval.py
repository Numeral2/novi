import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Streamlit UI za unos API ključeva i naziva indeksa
st.title("Retrieval Augmented Generation (RAG)")

# Unos API ključeva i naziva indeksa putem Streamlit UI
pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
index_name = st.text_input("Enter your Pinecone Index Name")

if pinecone_api_key and openai_api_key and index_name:
    # Inicijalizacija Pinecone-a sa korisnički unesenim API ključem
    pc = Pinecone(api_key=pinecone_api_key)

    # Inicijalizacija indeksa
    index = pc.Index(index_name)

    # Inicijalizacija OpenAI embeddings modela i Pinecone Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Postavljanje retriever-a
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )

    # Izvršavanje pretrage
    query = st.text_input("Ask a question", "What is retrieval augmented generation?")
    if query:
        results = retriever.invoke(query)

        # Prikazivanje rezultata
        st.subheader("Results:")
        for res in results:
            st.markdown(f"* {res.page_content} [{res.metadata}]")
else:
    st.warning("Please enter all required fields (API keys and index name).")

