# 📈 AI Financial Analyst (Serverless RAG System)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant_Cloud-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Gemini](https://img.shields.io/badge/LLM-Google_Gemini_2.5_Flash-orange)

<img width="901" height="845" alt="demo" src="https://github.com/user-attachments/assets/8e50bd40-7c67-407e-9665-aca114be8871" />

**[Try the Live App Here!](https://financialchatbot-hwlfgymxadd5gctck3xagx.streamlit.app/)**

## 📌 Overview
This project is a Retrieval-Augmented Generation (RAG) application designed to act as a specialized AI Financial Analyst. It ingests complex SEC 10-K financial reports, vectorizes the semantic content, and allows users to ask highly specific questions about a company's financial health, risk factors, and revenue streams.

## 🏗️ Cloud Architecture
* **Frontend/Hosting:** Hosted on Streamlit Community Cloud for a seamless, interactive chat interface.
* **Orchestration:** Built purely with **LangChain Expression Language (LCEL)** for readable, highly efficient data pipelines.
* **Vector Database:** Uses **Qdrant Cloud** to store embeddings remotely, keeping the repository lightweight and allowing for decoupled, asynchronous database updates.
* **Embedding & LLM:** Powered by Google's Gemini-1.0 embeddings and the Gemini 2.5 Flash chat model.

## 🚀 How It Works
The system is divided into two distinct pipelines:

1. **The Ingestion Pipeline (`ingest.py`):** Reads PDFs from the `/data` directory, chunks the text using a `RecursiveCharacterTextSplitter`, generates vector embeddings, and fires them directly to a remote Qdrant Cloud cluster in throttled batches.
2. **The Retrieval Pipeline (`query.py` & `app.py`):** A Streamlit interface takes user input, passes it through an LCEL chain to query the Qdrant Cloud database, retrieves the top 5 most relevant semantic chunks, and generates a factual response.

## 💻 Run It Locally

**1. Clone the repository**
```bash
git clone https://github.com/Folkk2/financial_chatbot.git
```

**2. Install Dependencies**
```bash
pip install -r requirments.txt
```

**3. Set up Environment Variables**
```bash
GOOGLE_API_KEY="your_google_api_key"
QDRANT_URL="your_qdrant_cloud_url"
QDRANT_API_KEY="your_qdrant_api_key"
```
**4. Run the app**
```bash
streamlit run app.py
```
## 🧠 Future Improvement
* Add multi-document comparison (e.g., comparing 2023 vs 2024 reports).
* Implement conversation memory (LangChain RunnableWithMessageHistory) for follow-up questions.

## 
