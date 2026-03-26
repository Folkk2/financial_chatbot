import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# load API
load_dotenv()

# Make the Chunk
def ingest_document():
    pdf_path = "./data" 
    
    loader = PyPDFDirectoryLoader(pdf_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split the document into {len(chunks)} searchable chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Qdrant Cloud Connection Secrets
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "sec_10k_reports"

    # Make Throttle for gemini free rate
    BATCH_SIZE = 90
    print("Starting throttled ingestion.")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        # View progress
        current_batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) // BATCH_SIZE) + 1
        print(f"Ingesting batch {current_batch_num} of {total_batches}...")
        
        # Add the batch directly to the Qdrant Cloud database
        QdrantVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            timeout=60.0
        )
        
        if i + BATCH_SIZE < len(chunks):
            print("Sleeping for 60 seconds to respect Google's free tier rate limits...")
            time.sleep(60)
    
    print("Bulk ingestion complete! The vector database is updated.")

if __name__ == "__main__":
    ingest_document()
