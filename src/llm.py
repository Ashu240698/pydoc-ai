"""
LLM Integration Module
Handles Groq API calls with conversation history and source citations.
"""

from groq import Groq
import config
from src.logger import logger
import time
from src.retrieval import RAGRetriever

class LLMHandler:
    """Handles LLM interactions with context from RAG."""

    def __init__(self, api_key):
        """
        Initialize the LLM handler with Groq client and RAG retriever.

        Sets up the necessary components for processing queries, including
        loading pre-built indexes for efficient retrieval.

        Args:
            api_key: Groq API key (if None, uses env variable)
        """

        # Use provided key or fall back to environment
        groq_key = api_key or config.GROQ_API_KEY

        if not groq_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")

        # Initialize Groq API client with API key from configuration
        self.client = Groq(api_key=groq_key)

        # Initialize RAG retriever for document retrieval
        self.retriever = RAGRetriever()

        # Load FAISS indexes immediately for performance
        # This ensures indexes are ready before any queries
        print("📂 Loading indexes...")
        self.retriever.load_indexes()
        print("✅ Indexes loaded and ready!")

        # Initialize conversation history as empty list
        self.conversation_history = []



    def build_prompt(self, query, retrieved_chunks):
        """
        Construct a comprehensive prompt for the LLM using retrieved context.

        This method formats the retrieved document chunks into a structured context
        that includes source citations, and builds both system and user prompts
        to guide the LLM's response generation.

        Args:
            query (str): The user's question or query.
            retrieved_chunks (list): List of dictionaries containing chunk text
                                   and metadata from the retrieval system.

        Returns:
            tuple: A pair of (system_prompt, user_prompt) strings for the LLM.
        """
        # Build context from retrieved chunks with source citations
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Add source header with citation number
            context_parts.append(f"[Source {i}] {chunk['metadata']['source']} - {chunk['metadata'].get('module', 'N/A')}")
            # Add the actual chunk text
            context_parts.append(chunk['text'])
            # Add spacing for readability
            context_parts.append("")  # Empty line for readability

        # Join all context parts into a single string
        context = "\n".join(context_parts)

        # Define system prompt with clear instructions for the AI assistant
        system_prompt = """You are PyDoc AI, a helpful Python documentation assistant.
                        Your job is to answer Python programming questions using the provided documentation context.

                        Guidelines:
                            1. Answer based ONLY on the provided context
                            2. Include code examples when relevant
                            3. Be concise and clear
                            4. If the context doesn't contain the answer, say so
                            5. Cite sources using [Source 1], [Source 2], etc.

                        Context from Python documentation:""" + context

        # Create user prompt with the original query
        user_prompt = f"Question: {query}"

        return system_prompt, user_prompt
        
    def query(self, user_query, k=4, include_history=True):
        """
        Process a user query through the complete RAG pipeline.

        This method orchestrates the entire query processing workflow:
        1. Retrieves relevant document chunks using RAG
        2. Builds contextual prompts for the LLM
        3. Generates response using Groq API
        4. Manages conversation history
        5. Extracts and formats source citations
        6. Logs performance metrics

        Args:
            user_query (str): The user's question to process.
            k (int, optional): Number of top chunks to retrieve. Defaults to 4.
            include_history (bool, optional): Whether to include conversation
                                             history in the prompt. Defaults to True.

        Returns:
            tuple: (response_text, sources) where response_text is the LLM's
                   answer and sources is a list of source metadata dictionaries.
        """
        # Record start time for performance monitoring
        start_time = time.time()

        # Display user query with visual separator
        print(f"\n{'='*70}")
        print(f"User: {user_query}")
        print('='*70)

        # Step 1: Retrieve relevant document chunks using RAG system
        retrieved = self.retriever.retrieve(user_query, k_final=k)

        # Step 2: Build structured prompts with retrieved context
        system_prompt, user_prompt = self.build_prompt(user_query, retrieved)

        # Step 3: Construct messages array for API call
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # Include conversation history if enabled
        if include_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        # Step 4: Call Groq API for response generation
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )

        # Extract response text from API response
        response_text = response.choices[0].message.content

        # Step 5: Update conversation history if enabled
        if include_history:
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Trim history to prevent excessive memory usage
            if len(self.conversation_history) > config.MAX_CONVERSATION_HISTORY:
                self.conversation_history = self.conversation_history[-config.MAX_CONVERSATION_HISTORY:]

        # Step 6: Extract and format source metadata for citations
        sources = [
            {
                'source': chunk['metadata']['source'],
                'module': chunk['metadata'].get('module', 'N/A'),
                'url': chunk['metadata']['url'],
                'score': chunk.get('rerank_score', chunk.get('hybrid_score', 0))
            }
            for chunk in retrieved
        ]

        # Step 7: Log query details and performance metrics
        duration = time.time() - start_time
        logger.log_query(
            query=user_query,
            num_results=len(retrieved),
            response_length=len(response_text),
            sources=sources
        )
        logger.log_performance('full_query_pipeline', duration)

        return response_text, sources
        
    def display_response(self, response_text, sources):
        """
        Display the LLM response and source citations in a formatted manner.

        This method provides a clean, readable output format for the assistant's
        response and lists all sources with their relevance scores and URLs.

        Args:
            response_text (str): The LLM-generated response to display.
            sources (list): List of source metadata dictionaries to cite.
        """
        # Display the assistant's response with visual separator
        print(f"\n{'='*70}")
        print("PyDoc AI:")
        print('='*70)
        print(response_text)

        # Display sources section with detailed metadata
        print(f"\n{'='*70}")
        print("📚 SOURCES")
        print('='*70)

        for i, source in enumerate(sources, 1):
            print(f"\n[{i}] {source['source']} - {source['module']}")
            print(f"    Score: {source['score']:.2f}")
            print(f"    URL: {source['url']}")
    
    def chat(self):
        """
        Run an interactive chat loop for real-time user interaction.

        This method provides a command-line interface where users can ask
        questions about Python documentation. It supports special commands
        for quitting, clearing history, and handles empty inputs gracefully.
        """
        # Display welcome message and instructions
        print("\n" + "="*70)
        print("🐍 PyDoc AI - Python Documentation Assistant")
        print("="*70)
        print("Ask me anything about Python!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to reset conversation history.")
        print("="*70 + "\n")

        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Skip empty inputs
            if not user_input:
                continue

            # Handle quit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break

            # Handle clear history command
            if user_input.lower() == 'clear':
                self.conversation_history = []
                print("\n✅ Conversation history cleared!")
                continue

            # Process the query and get response
            response_text, sources = self.query(user_input)

            # Display the formatted response
            self.display_response(response_text, sources)


def main():
    """
    Main entry point for testing the LLM handler functionality.

    This function demonstrates the basic usage of the LLMHandler class
    by running a sample query about reading CSV files in Python.
    """
    # Initialize the LLM handler
    handler = LLMHandler()

    # Define a test query
    query = "How do I read a CSV file in Python?"

    # Process the query and get response
    response, sources = handler.query(query)

    # Display the results
    handler.display_response(response, sources)


if __name__ == "__main__":
    # Execute main function when script is run directly
    main()

            