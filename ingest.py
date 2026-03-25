import os
import time
from dotenv import load_dotenv

#langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Load environment variables (your API key)
load_dotenv()

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

    print("Generating embeddings and saving to ChromaDB")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    BATCH_SIZE = 90
    print("Starting throttled ingestion. Grab a coffee, this will take a few minutes! ☕")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        # Calculate progress
        current_batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) // BATCH_SIZE) + 1
        print(f"Ingesting batch {current_batch_num} of {total_batches}...")
        
        # Add the batch to the database
        vector_store.add_documents(batch)
        
        # If we are not on the very last batch, sleep to reset the API quota
        if i + BATCH_SIZE < len(chunks):
            print("Sleeping for 60 seconds to respect Google's free tier rate limits...")
            time.sleep(60)
    
    print("Bulk ingestion complete! The vector database is updated.")

if __name__ == "__main__":
    ingest_document()
