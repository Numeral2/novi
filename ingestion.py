import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Učitaj .env varijable
load_dotenv()

# Inicijalizacija Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")  # Promijeni ako je potrebno

# Provjera i kreiranje indeksa ako ne postoji
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Postavi odgovarajući broj dimenzija prema embedding modelu
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Inicijalizacija embedding modela i Pinecone Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Učitaj PDF-ove
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# Postavke chunkanja
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# Procesiranje dokumenata
chunk_count = 0  # Brojač chunkova
for doc in raw_documents:
    chunks = text_splitter.split_text(doc.page_content)
    
    for chunk_index, chunk in enumerate(chunks):
        chunk_count += 1
        chunk_metadata = {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "chunk_id": f"{chunk_count}",
            "text_snippet": chunk[:50]  # Prvih 50 znakova kao pregled
        }
        document = Document(page_content=chunk, metadata=chunk_metadata)
        
        # Dodaj pojedinačni chunk u Pinecone
        vector_store.add_documents(documents=[document], ids=[f"id{chunk_count}"])
        print(f"✅ Dodan chunk {chunk_count} u Pinecone.")

print(f"✅ Ukupno dodano {chunk_count} chunkova u Pinecone.")

