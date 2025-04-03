# Multi-Agentic Retrieval System
 
## Description
This system enables retrieval and ranking of documents based on user queries. It uses a multi-agent approach involving a query parser, document retriever, document ranker, and a text generation model (GPT-2). The system can handle document-based question answering and ranking with high accuracy and speed.
 
## Features
- Query parsing to extract keywords.
- Document retrieval and ranking using embeddings.
- Response generation based on the context and query using GPT-2.
- Streamlit interface for easy interaction.

## **Results Preview** 🖼️

  ![Screenshot (2)](https://github.com/user-attachments/assets/81871c97-428b-4484-91f7-74822ca66edd)

  this is the image where we basically paste our query this is how it gives us the output below is the image of the output
  
   ![Screenshot (4)](https://github.com/user-attachments/assets/761cf897-8154-4edf-a98f-6e77d69957ed)

## Setup Instructions
 
### Prerequisites
1. Python 3.x
2. Install the required libraries:
 
```bash
pip install streamlit transformers sentence-transformers numpy

Myenv\Scripts\activate

streamlit run app.py




