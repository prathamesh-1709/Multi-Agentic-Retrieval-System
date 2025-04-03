# Multi-Agentic Retrieval System
 
## Description
This system enables retrieval and ranking of documents based on user queries. It uses a multi-agent approach involving a query parser, document retriever, document ranker, and a text generation model (GPT-2). The system can handle document-based question answering and ranking with high accuracy and speed.
 
## Features
- Query parsing to extract keywords.
- Document retrieval and ranking using embeddings.
- Response generation based on the context and query using GPT-2.
- Streamlit interface for easy interaction.
 
## Setup Instructions
 
### Prerequisites
1. Python 3.x
2. Install the required libraries:
 
```bash
pip install streamlit transformers sentence-transformers numpy

Myenv\Scripts\activate

streamlit run app.py