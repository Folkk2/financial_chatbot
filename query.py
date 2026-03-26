import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="sec_10k_reports",
    url=qdrant_url,
    api_key=qdrant_api_key
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

system_prompt = (
    "You are an expert Financial Analyst Assistant. "
    "Use the following pieces of retrieved context from SEC 10-K filings to answer the user's question. "
    "If you don't know the answer, just say that you don't know. "
    "Do NOT make up information.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_database(query):
    print(f"\nSearching database for: '{query}'...")
    
    answer = rag_chain.invoke(query)
    
    return answer

if __name__ == "__main__":
    user_question = "What are the primary risk factors mentioned in the report?"
    answer = ask_database(user_question)
    
    print("\n--- AI Response ---")
    print(answer)
    print("-------------------")