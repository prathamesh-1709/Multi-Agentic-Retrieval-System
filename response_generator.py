import openai  # Import OpenAI library for interacting with the OpenAI API.
 
class ResponseGenerator:
    def __init__(self, api_key):
        """
        Initializes the ResponseGenerator object with the provided API key.
        Parameters:
        - api_key: The API key for authenticating with OpenAI.
        """
        openai.api_key = api_key  # Set the API key for OpenAI authentication.
 
    def generate_response(self, context, query):
        """
        Generates a response from OpenAI's GPT model based on the context and query.
        Parameters:
        - context: The context in which the query is made, typically the retrieved documents.
        - query: The user's input question or query.
        Returns:
        - A string containing the model's generated response.
        """
        # Using the updated API with openai.Completion.create to generate a response.
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Specify the model to use (e.g., gpt-4).
            prompt=f"Context: {context}\nQuery: {query}",  # Formulate the prompt by combining context and query.
            max_tokens=150  # Limit the response to 150 tokens (you can adjust this as needed).
        )
        # Extract the generated response from the API response and return it.
        return response.choices[0].text.strip()

