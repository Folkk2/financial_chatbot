import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Connect to Database & Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Create the Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# 5. Define the Prompt Template
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

# 6. Helper function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------------
# 7. THE MODERN WAY: Pure LCEL (LangChain Expression Language)
# This is the exact data flow: Dictionary -> Prompt -> LLM -> String Output
# ---------------------------------------------------------
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Create a reusable function to ask questions
def ask_database(query):
    print(f"\nSearching database for: '{query}'...")
    
    # We invoke the chain directly with our query string
    answer = rag_chain.invoke(query)
    
    return answer

# --- Testing the Code ---
if __name__ == "__main__":
    user_question = "What are the primary risk factors mentioned in the report?"
    answer = ask_database(user_question)
    
    print("\n--- AI Response ---")
    print(answer)
    print("-------------------")