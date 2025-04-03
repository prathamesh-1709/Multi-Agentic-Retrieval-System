import spacy  
# Import the spaCy library, which is used for natural language processing.
 
class Query_parser:
    def __init__(self):
        # Initialize the QueryParser class.
        # When an object of QueryParser is created, it loads a pre-trained spaCy model called "en_core_web_sm"
        # This model will be used to process and understand English text.
        self.nlp = spacy.load("en_core_web_sm")
 
    def parse_query(self, query):
        """
        This function takes a query (a sentence or question in string form) as input,
        processes it using the spaCy model, and returns a list of relevant words.
 
        The steps:
        1. The query is passed to the spaCy model, which analyzes the text and creates a document object.
        2. The function iterates through each word (token) in the document.
        3. It checks if the token is an alphabetic word (`token.is_alpha`) and not a stop word (`token.is_stop`).
        4. Only words that meet these conditions are kept, and their text (`token.text`) is returned in a list.
        """
        # Pass the query to spaCy's model to process the text and create a "document" object.
        doc = self.nlp(query)
        # Iterate through each token (word) in the document and filter out stop words and non-alphabetical tokens.
        # Stop words are common words (like "the", "is", "on", etc.) that don't carry important meaning for query parsing.
        return [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]