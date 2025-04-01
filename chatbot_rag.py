import streamlit as st
import os
import pdfplumber  # For extracting text from PDF
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

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

    # Helper function to chunk text into smaller parts
    def chunk_text(text, max_size=4000):
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
            text = ""
            for page in pdf.pages:
                # Extract text while ignoring tables and images
                text += page.extract_text() or ""  # Only append if text is available

        # Chunk the text
        chunks = chunk_text(text)

        # Store each chunk in Pinecone with metadata (e.g., chunk index)
        for i, chunk in enumerate(chunks):
            # Get the embedding of the chunk
            embedding = embeddings.embed_documents([chunk])[0]
            
            # Create a metadata dictionary for each chunk
            metadata = {"chunk_index": i, "content": chunk}

            # Upsert into Pinecone
            index.upsert(
                vectors=[{
                    "id": str(i),
                    "values": embedding,
                    "metadata": metadata
                }],
                namespace="pdf_chunks"
            )

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

        # Configure the retriever to fetch relevant document chunks
        response = index.query(
            namespace="pdf_chunks",
            vector=embeddings.embed_documents([prompt])[0],
            top_k=3,
            include_values=True,
            include_metadata=True,
        )

        # Collect relevant document chunks from the query result
        docs_text = "\n".join([item["metadata"]["content"] for item in response["matches"]])

        # Define the system prompt to generate a response based on the context
        system_prompt = f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Context: {docs_text}"""

        # Create the assistant's response using OpenAI's API
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=1, openai_api_key=openai_api_key)
        result = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]).content

        # Display the assistant's answer
        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

else:
    st.warning("Please enter your API keys and index name.")

