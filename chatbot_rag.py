import streamlit as st
import os
from dotenv import load_dotenv
import pdfplumber  # For extracting text from PDF
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone.core import Vector

load_dotenv()

st.title("Chatbot with PDF Upload")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired
index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# helper function to chunk text into smaller parts
def chunk_text(text, max_size=40000):
    chunks = []
    current_chunk = ""
    
    # Split the text into smaller chunks to keep the size under the limit
    for line in text.split("\n"):
        if len(current_chunk) + len(line) <= max_size:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())  # Append any remaining content
    
    return chunks

# handle file upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # Read the uploaded PDF using pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Chunk the extracted text
    chunks = chunk_text(text)

    # Store each chunk in Pinecone
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed(chunk)
        vector = Vector(id=str(i), values=embedding)
        index.upsert([vector])  # Insert into Pinecone

    st.success("PDF has been processed and stored in Pinecone.")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Ask me something")

# did the user submit a prompt?
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # initialize the llm
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=1
    )

    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # creating the system prompt
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}:"""

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    # adding the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # invoking the llm
    result = llm.invoke(st.session_state.messages).content

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))
