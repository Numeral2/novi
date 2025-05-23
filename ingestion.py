import os
import time
import pdfplumber  # Using pdfplumber for PDF text extraction

# Import Pinecone and Langchain
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Get user input for Pinecone API key and index name
pinecone_api_key = input("Enter your Pinecone API Key: ")
openai_api_key = input("Enter your OpenAI API Key: ")
index_name = input("Enter your Pinecone Index Name: ")

# Initialize Pinecone with the user-provided API key
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, and create it if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # This should match the model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Get the Pinecone index
index = pc.Index(index_name)

# Initialize OpenAI embeddings model and Pinecone vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# Always use "pdf_chunks" as namespace
namespace = "pdf_chunks"

# Initialize Pinecone vector store with specified namespace
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

# Get user input for the directory containing PDFs
pdf_directory_path = input("Enter the directory path containing your PDFs: ")

# Function to extract text from PDFs using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract only text, ignoring images and tables
        return text

# Load PDFs from the specified directory
raw_documents = []
for filename in os.listdir(pdf_directory_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory_path, filename)
        text = extract_text_from_pdf(pdf_path)
        
        if text.strip():  # Ensure the extracted text is not empty
            raw_documents.append(text)

# Check if documents are being loaded correctly
if not raw_documents:
    print("No documents loaded. Please check the directory path.")
else:
    print(f"{len(raw_documents)} documents loaded successfully.")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Adjust based on your document size
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# Split the documents into chunks
chunks = text_splitter.split_documents(raw_documents)

# Check if chunks are being generated
if not chunks:
    print("No chunks generated. Please check the text splitting logic.")
else:
    print(f"{len(chunks)} chunks generated.")

# Add chunks to Pinecone with metadata
for i, chunk in enumerate(chunks):
    metadata = {"source": f"Document {i}", "chunk_index": i}  # Customize metadata as needed
    document = Document(page_content=chunk, metadata=metadata)
    
    # Always store in "pdf_chunks" namespace
    vector_store.add_documents(documents=[document], ids=[f"id_{i}"], namespace=namespace)

print(f"Added {len(chunks)} chunks to Pinecone in namespace '{namespace}'.")

