# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# loading the PDF document
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# function to split text based on byte size
def split_by_bytes(text, max_bytes=40000, overlap_bytes=2000):
    chunks = []
    start = 0
    encoded_text = text.encode("utf-8")  # Convert to bytes

    while start < len(encoded_text):
        end = start + max_bytes
        chunk = encoded_text[start:end].decode("utf-8", errors="ignore")  # Decode safely
        chunks.append(chunk)
        start += max_bytes - overlap_bytes  # Overlap to maintain context

    return chunks

# process documents
final_documents = []
doc_id = 1

for doc in raw_documents:
    chunks = split_by_bytes(doc.page_content)  # Split into valid byte-sized chunks
    for chunk in chunks:
        final_documents.append(Document(page_content=chunk, metadata=doc.metadata))

# generate unique ids
uuids = [f"id{i}" for i in range(1, len(final_documents) + 1)]

# add to database
vector_store.add_documents(documents=final_documents, ids=uuids)

print(f"Successfully added {len(final_documents)} chunks to Pinecone.")

