import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")  # Set the index name

# Check if the index exists, and create it if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust dimension based on your model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load PDFs from the specified directory
loader = PyPDFDirectoryLoader("path_to_documents/")
raw_documents = loader.load()

# Set up the text splitter to create manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Adjust chunk size based on your document
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# Process documents into chunks and add to Pinecone
chunk_count = 0
for doc in raw_documents:
    chunks = text_splitter.split_text(doc.page_content)
    
    for chunk_index, chunk in enumerate(chunks):
        chunk_count += 1
        chunk_metadata = {
            "source": doc.metadata.get("source", "unknown"),  # Add source info
            "page": doc.metadata.get("page", "unknown"),      # Add page number
            "chunk_id": f"{chunk_count}",                      # Unique chunk ID
            "text_snippet": chunk[:50]                          # First 50 characters for preview
        }
        
        # Create a Document with the chunk and metadata
        document = Document(page_content=chunk, metadata=chunk_metadata)
        
        # Add the document chunk to Pinecone with a unique ID
        vector_store.add_documents(documents=[document], ids=[f"id{chunk_count}"])
        print(f"✅ Added chunk {chunk_count} to Pinecone.")

print(f"✅ Total {chunk_count} chunks added to Pinecone.")

