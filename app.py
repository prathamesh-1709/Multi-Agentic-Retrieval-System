# import streamlit as st
# from transformers import pipeline
# from query_parser import Query_parser  # Import the QueryParser class
# from document_retrieval import DocumentRetriever  # Import the DocumentRetriever class
# from document_ranker import DocumentRanker  # Import the DocumentRanker class
 
# # Initialize the Hugging Face text generation pipeline with GPT-2 model
# generator = pipeline("text-generation", model="gpt2")
 
# # Initialize the components of the multi-agentic system
# parser = Query_parser()  # Initialize QueryParser
# retriever = DocumentRetriever(doc_dir="documents/")  # Initialize DocumentRetriever
# ranker = DocumentRanker()  # Initialize DocumentRanker
 
# # Set the title of the Streamlit app
# st.title("Multi-Agentic Retrieval System")
 
# # Input field for user query
# query = st.text_input("Enter your query")
 
# # If query is entered, process it
# if query:
#     # Parse the query to extract keywords
#     keywords = parser.parse_query(query)
#     st.write("Keywords:", keywords)
 
#     # Retrieve documents based on parsed keywords
#     results = retriever.search(" ".join(keywords))
#     st.write("Top Documents:", results)
 
#     # Rank the documents by relevance
#     ranked_docs = ranker.rank(results)
#     st.write("Top Ranked Documents:", ranked_docs)
 
#     # Create context by combining content of top 3 ranked documents
#     try:
#         # Safeguard against missing files and ensure text size is manageable
#         context = " ".join(
#             [open(f"documents/{doc[0]}").read() for doc in ranked_docs[:3]]
#         )
#         context = context[:1000]  # Truncate context to a safe length
#     except FileNotFoundError as e:
#         st.error(f"Error reading document: {e}")
#         context = ""
 
#     if context.strip():  # Proceed only if the context is valid
#         # Generate a response based on the context and query using Hugging Face
#         input_text = f"Context: {context}\nQuery: {query}"
#         response = generator(input_text, max_new_tokens=50, num_return_sequences=1)
 
#         # Display the generated response
#         st.write("Response:", response[0]["generated_text"])
#     else:
#         st.error("Context could not be generated due to missing documents.")



# import streamlit as st
# from transformers import pipeline
# from query_parser import Query_parser  # Import the QueryParser class
# from document_retrieval import DocumentRetriever  # Import the DocumentRetriever class
# from document_ranker import DocumentRanker  # Import the DocumentRanker class
 
# # Initialize the Hugging Face text generation pipeline with GPT-2 model
# generator = pipeline("text-generation", model="gpt2")
 
# # Initialize the components of the multi-agentic system
# parser = Query_parser()  # Initialize QueryParser
# retriever = DocumentRetriever(doc_dir="documents/")  # Initialize DocumentRetriever
# ranker = DocumentRanker()  # Initialize DocumentRanker
 
# # Set the title of the Streamlit app
# st.title("Multi-Agentic Retrieval System")
 
# # Input field for user query
# query = st.text_input("Enter your query")
 
# # If query is entered, process it
# if query:
#     # Parse the query to extract keywords
#     keywords = parser.parse_query(query)
#     st.write("Keywords:", keywords)
 
#     # Retrieve documents based on parsed keywords
#     results = retriever.search(" ".join(keywords))
#     st.write("Top Documents:", results)
 
#     # Rank the documents by relevance
#     ranked_docs = ranker.rank(results)
#     st.write("Top Ranked Documents:", ranked_docs)
 
#     # Create context by combining content of top 3 ranked documents
#     try:
#         # Safeguard against missing files and ensure text size is manageable
#         context = " ".join(
#             [open(f"documents/{doc[0]}").read() for doc in ranked_docs[:3]]
#         )
#         context = context[:1000]  # Truncate context to a safe length
#     except FileNotFoundError as e:
#         st.error(f"Error reading document: {e}")
#         context = ""
 
#     if context.strip():  # Proceed only if the context is valid
#         # Generate a response based on the context and query using Hugging Face
#         input_text = f"Context: {context}\nQuery: {query}"
#         response = generator(input_text, max_new_tokens=50, num_return_sequences=1)
 
#         # Display the generated response
#         st.write("Response:", response[0]["generated_text"])
#     else:
#         st.error("Context could not be generated due to missing documents.")

import streamlit as st
from transformers import pipeline
from query_parser import Query_parser  # Import the QueryParser class
from document_retrieval import DocumentRetriever  # Import the DocumentRetriever class
from document_ranker import DocumentRanker  # Import the DocumentRanker class
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Hugging Face text generation pipeline with GPT-3.5
generator = pipeline("text-generation", model="gpt2")

# Initialize the components of the multi-agentic system
parser = Query_parser()
retriever = DocumentRetriever(doc_dir="documents/")
ranker = DocumentRanker()

# Set the title of the Streamlit app
st.title("Multi-Agentic Retrieval System")

# Input field for user query
query = st.text_input("Enter your query")

# If query is entered, process it
if query:
    if not query.strip():
        st.error("Please enter a valid query!")
    else:
        # Parse the query to extract keywords
        keywords = parser.parse_query(query)
        st.write("Keywords:", keywords)
        if not keywords:
            st.error("No relevant keywords extracted from the query!")
        else:
            # Retrieve documents based on parsed keywords
            results = retriever.search(" ".join(keywords))
            st.write("Top Documents:", results)

            # Rank the documents by relevance
            ranked_docs = ranker.rank(results, keywords)
            st.write("Top Ranked Documents:", ranked_docs)

            # Create context by combining content of top 3 ranked documents
            try:
                context = " ".join(
                    [open(f"documents/{doc[0]}").read() for doc in ranked_docs[:3]]
                )
                context = context[:1000]  # Truncate context to a safe length
            except FileNotFoundError as e:
                st.error(f"Error reading document: {e}")
                context = ""

            if context.strip():
                # Generate a response based on the context and query using Hugging Face
                input_text = f"Context: {context}\nQuery: {query}"
                response = generator(input_text, max_new_tokens=50, num_return_sequences=1)
                st.write("Response:", response[0]["generated_text"])
            else:
                st.error("Context could not be generated due to missing documents.")

