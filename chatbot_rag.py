import streamlit as st
import os
import pdfplumber  # For extracting text from PDF
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone

# Streamlit UI for user input (API keys and index name)
st.title("Chatbot with PDF Upload")

# User inputs for API keys
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")
pinecone_index_name = st.text_input("Enter your Pinecone Index Name:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Initialize Pinecone database only if keys are provided
if pinecone_api_key and pinecone_index_name:
    # Initialize Pinecone with the API key provided by the user
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    # Initialize embeddings model + vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Helper function to chunk text into smaller parts
    def chunk_text(text, max_size=40000):
        chunks = []
        current_chunk = ""

        for line in text.split("\n"):
            if len(current_chunk) + len(line) <= max_size:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Handle file upload
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        # Chunk the text
        chunks = chunk_text(text)

        # Store each chunk in Pinecone
        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_documents([chunk])[0]
            # Add chunk metadata for better retrieval
            index.upsert([(str(i), embedding, {"chunk_index": i, "content": chunk})])

        st.success("PDF has been processed and stored in Pinecone.")

    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "system", "content": "You are an assistant for question-answering tasks."})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input field
    prompt = st.chat_input("Ask me something")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        # Use the OpenAI model for the response
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=1, openai_api_key=openai_api_key)

        # Configure the retriever to fetch relevant document chunks
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})
        docs = retriever.invoke(prompt)
        docs_text = "\n".join([d.page_content for d in docs])

        # Define the system prompt to generate a response based on the context
        system_prompt = f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Context: {docs_text}"""

        st.session_state.messages.append({"role": "system", "content": system_prompt})

        # Get the assistant's response
        result = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]).content

        # Display the assistant's answer
        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
else:
    st.warning("Please enter your API keys and index name.")

