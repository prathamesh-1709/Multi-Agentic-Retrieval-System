# import os
# import faiss
# from sentence_transformers import SentenceTransformer
# import numpy as np

# class DocumentRetriever:
#     def __init__(self, doc_dir):
#         """
#         Initializes the DocumentRetriever object.
#         Parameters:
#         - doc_dir: The directory where the documents (text files) are stored.
#         """
#         # Load the pre-trained sentence transformer model 'all-MiniLM-L6-v2'.
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         # Store the directory of documents.
#         self.doc_dir = doc_dir
#         # Initialize FAISS index for efficient similarity search (L2 distance).
#         self.index = faiss.IndexFlatL2(384)  # 384 is the dimension of the embeddings from 'all-MiniLM-L6-v2'
#         # A mapping from index positions to document filenames.
#         self.doc_map = []
#         # Build the FAISS index using the documents in the given directory.
#         self._build_index()

#     def _build_index(self):
#         """
#         Builds the FAISS index by reading documents from the directory,
#         encoding them into embeddings, and adding them to the FAISS index.
#         """
#         # Initialize an array to store the embeddings of documents.
#         embeddings = []

#         # Iterate through each file in the document directory.
#         for idx, filename in enumerate(os.listdir(self.doc_dir)):
#             # Open the file and read its content using 'utf-8' encoding to avoid errors.
#             try:
#                 with open(os.path.join(self.doc_dir, filename), 'r', encoding='utf-8') as f:
#                     text = f.read()  # Read the full text from the file.
#                     # Encode the document text into an embedding (vector) using the model.
#                     embedding = self.model.encode(text)
#                     embeddings.append(embedding)  # Add the embedding to the list.
#                     # Map the index to the document filename for retrieval.
#                     self.doc_map.append(filename)
#             except UnicodeDecodeError:
#                 print(f"Could not read file {filename} due to encoding issues.")

#         # Convert the list of embeddings into a numpy array.
#         embeddings = np.array(embeddings)

#         # Normalize the embeddings for FAISS (important for L2 distance).
#         faiss.normalize_L2(embeddings)

#         # Add the embeddings to the FAISS index.
#         self.index.add(embeddings)

#     def search(self, query, top_k=5):
#         """
#         Search for the most relevant documents based on the given query.
#         Parameters:
#         - query: The user's input query (sentence or question).
#         - top_k: The number of top documents to return (default is 5).
#         Returns:
#         - A list of tuples with the document filename and its similarity distance to the query.
#         """
#         # Encode the query into an embedding (vector).
#         query_embedding = self.model.encode(query)
        
#         # Normalize the query embedding (important for FAISS with L2 distance).
#         faiss.normalize_L2(query_embedding)

#         # Perform a search in the FAISS index to find the top-k closest documents to the query embedding.
#         distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

#         # Check if we have enough results (FAISS may return fewer results than top_k)
#         if len(distances[0]) < top_k:
#             print(f"Warning: Found only {len(distances[0])} results instead of {top_k}.")

#         # Return the top-k document filenames along with their respective similarity distances.
#         return [(self.doc_map[idx], distances[0][i]) for i, idx in enumerate(indices[0])]



import os  # Provides a way to interact with the operating system (e.g., file handling).
import faiss  # FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering.
from sentence_transformers import SentenceTransformer  # Sentence Transformers is a library for sentence embeddings.
 
class DocumentRetriever:
    def __init__(self, doc_dir):
        """
        Initializes the DocumentRetriever object.
        Parameters:
        - doc_dir: The directory where the documents (text files) are stored.
        """
        # Load the pre-trained sentence transformer model 'all-MiniLM-L6-v2'.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Store the directory of documents.
        self.doc_dir = doc_dir
        # Initialize FAISS index for efficient similarity search (L2 distance).
        # FAISS IndexFlatL2 is used for flat (brute-force) search with L2 distance metric.
        self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())  # 384 is the dimension of the embeddings from 'all-MiniLM-L6-v2'
        # A mapping from index positions to document filenames.
        self.doc_map = []
        # Build the FAISS index using the documents in the given directory.
        self._build_index()
 
    def _build_index(self):
        """
        Builds the FAISS index by reading documents from the directory,
        encoding them into embeddings, and adding them to the FAISS index.
        """
        # Iterate through each file in the document directory.
        for idx, filename in enumerate(os.listdir(self.doc_dir)):
            # Open the file and read its content.
            with open(os.path.join(self.doc_dir, filename), 'r',encoding='utf-8') as f:
                text = f.read()  # Read the full text from the file.
                # Encode the document text into an embedding (vector) using the model.
                embedding = self.model.encode(text)
                # Add the document's embedding to the FAISS index.
                self.index.add(embedding.reshape(1, -1))  # Reshaping for FAISS compatibility (1, embedding_dim).
                # Map the index to the document filename for retrieval.
                self.doc_map.append(filename)
 
    # def search(self, query, top_k=5):
    #     """
    #     Search for the most relevant documents based on the given query.
    #     Parameters:
    #     - query: The user's input query (sentence or question).
    #     - top_k: The number of top documents to return (default is 5).
    #     Returns:
    #     - A list of tuples with the document filename and its similarity distance to the query.
    #     """

    #     if self.index.ntotal == 0:
    #         raise ValueError("FAISS index is empty. No documents were added.")
    #     # Encode the query into an embedding (vector)
    #     query_embedding = self.model.encode(query)
    #     # Perform a search in the FAISS index to find the top-k closest documents to the query embedding.
    #     distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
    #     # Return the top-k document filenames along with their respective similarity distances.
    #     return [(self.doc_map[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    ''' the above code is working so we can use that as well just this is the updated one'''

    def search(self, query, top_k=5):
        """
        Search for the most relevant documents based on the given query, 
        combining FAISS similarity and keyword matching.
        Parameters:
        - query: The user's input query (sentence or question).
        - top_k: The number of top documents to return (default is 5).
        Returns:
        - A list of tuples with the document filename and its adjusted score.
        """

        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. No documents were added.")
        
        # Encode the query into an embedding (vector)
        query_embedding = self.model.encode(query)

        # Perform a FAISS search to find the top-k closest documents
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        # Gather the retrieved documents
        retrieved_docs = [(self.doc_map[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

        # Split the query into individual words (basic tokenization)
        query_words = set(query.lower().split())

        # Create a list to hold documents with adjusted scores
        adjusted_docs = []

        # Check each document for keyword matches and adjust the score
        for doc_filename, distance in retrieved_docs:
            with open(os.path.join(self.doc_dir, doc_filename), 'r', encoding='utf-8') as f:
                doc_content = f.read().lower()
                keyword_match = any(word in doc_content for word in query_words)
                
                # Adjust the score: lower distance for keyword matches
                adjusted_score = distance - 0.5 if keyword_match else distance
                adjusted_docs.append((doc_filename, adjusted_score))

        # Sort documents by adjusted score (lower scores = more relevant)
        adjusted_docs = sorted(adjusted_docs, key=lambda x: x[1])

        # Return the top-k adjusted documents
        return adjusted_docs[:top_k]


        