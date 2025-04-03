class DocumentRanker:
    def rank(self, retrieved_docs,query_words):
        """
        This method ranks the retrieved documents based on their relevance to the query.
        Parameters:
        - retrieved_docs: A list of tuples where each tuple contains a document filename
          and its similarity score (distance) to the query.
        Returns:
        - A list of documents sorted in descending order of relevance (i.e., lowest distance first).
        """
        # Sort the retrieved documents by the second element in each tuple (the distance),
        # in descending order (higher similarity first).
                # Adjust the score based on keyword presence
        ''' this thing is added later'''
        ranked_docs = []
        for doc_filename, distance in retrieved_docs:
            # Check if any query words are present in the document filename
            keyword_match = any(word.lower() in doc_filename.lower() for word in query_words)
            # Apply a penalty or boost for keyword matches (adjust the distance)
            adjusted_score = distance - 0.5 if keyword_match else distance
            ranked_docs.append((doc_filename, adjusted_score))

        # Sort the retrieved documents by adjusted score (lowest distance = highest relevance)
        return sorted(ranked_docs, key=lambda x: x[1])
        # return sorted(retrieved_docs, key=lambda x: x[1],reverse=True)
