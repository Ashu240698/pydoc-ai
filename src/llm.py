"""
LLM Integration Module
Handles Groq API calls with conversation history and source citations.
"""

from groq import Groq
import config
from src.retrieval import RAGRetriever

class LLMHandler:
    """Handles LLM interactions with context from RAG."""

    def __init__(self):
        """Initialize Groq client."""
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.retriever = RAGRetriever()
        # Load indexes immediately (ADD THIS!)
        print("📂 Loading indexes...")
        self.retriever.load_indexes()
        print("✅ Indexes loaded and ready!")
        self.conversation_history = []



    def build_prompt(self, query, retrieved_chunks):
        """
            Build prompt with retrieved context.
        
            Args:
                query: User question
                retrieved_chunks: List of dicts with chunk text and metadata
        
            Returns:
                system_prompt: System instructions
                user_prompt: User query with context
        """

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}] {chunk['metadata']['source']} - {chunk['metadata'].get('module', 'N/A')}")
            context_parts.append(chunk['text'])
            context_parts.append("")  # Empty line for readability
            
        context = "\n".join(context_parts)
        
        # System prompt
        system_prompt = """You are PyDoc AI, a helpful Python documentation assistant.
                        Your job is to answer Python programming questions using the provided documentation context.

                        Guidelines:
                            1. Answer based ONLY on the provided context
                            2. Include code examples when relevant
                            3. Be concise and clear
                            4. If the context doesn't contain the answer, say so
                            5. Cite sources using [Source 1], [Source 2], etc.

                        Context from Python documentation:""" + context
        # User prompt
        user_prompt = f"Question: {query}"
        
        return system_prompt, user_prompt
        
    def query(self, user_query, k=4, include_history=True):
        """
            Process a user query with RAG pipeline.
        
            Args:
                user_query: User's question
                k: Number of chunks to retrieve
                include_history: Whether to include conversation history
        
            Returns:
                response: LLM response text
                sources: List of source metadata
            """
        
        print(f"\n{'='*70}")
        print(f"User: {user_query}")
        print('='*70)
        
        # Step 1: Retrieve relevant chunks
        retrieved = self.retriever.retrieve(user_query, k_final=k)
        
        # Step 2: Build prompt
        system_prompt, user_prompt = self.build_prompt(user_query, retrieved)

        messages = []
        messages.append({"role":"system", "content":system_prompt})

        if include_history:
            messages.extend(self.conversation_history)

        messages.append({"role":"user", "content":user_prompt})

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )

        response_text = response.choices[0].message.content

        if include_history:
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": response_text})

        if len(self.conversation_history) > config.MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-config.MAX_CONVERSATION_HISTORY:]

        # Step 6: Extract sources
        sources = [
                {
                    'source': chunk['metadata']['source'],
                    'module': chunk['metadata'].get('module', 'N/A'),
                    'url': chunk['metadata']['url'],
                    'score': chunk.get('rerank_score', chunk.get('hybrid_score', 0))
                }
                for chunk in retrieved
            ]
        
        return response_text, sources
        
    def display_response(self, response_text, sources):
        """Display response and sources nicely."""
        
        print(f"\n{'='*70}")
        print("PyDoc AI:")
        print('='*70)
        print(response_text)
        
        print(f"\n{'='*70}")
        print("📚 SOURCES")
        print('='*70)
        
        for i, source in enumerate(sources, 1):
            print(f"\n[{i}] {source['source']} - {source['module']}")
            print(f"    Score: {source['score']:.2f}")
            print(f"    URL: {source['url']}")
    
    def chat(self):
        """Interactive chat loop."""
        
        print("\n" + "="*70)
        print("🐍 PyDoc AI - Python Documentation Assistant")
        print("="*70)
        print("Ask me anything about Python!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to reset conversation history.")
        print("="*70 + "\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                self.conversation_history = []
                print("\n✅ Conversation history cleared!")
                continue
            
            # Get response
            response_text, sources = self.query(user_input)
            
            # Display
            self.display_response(response_text, sources)


def main():
    """Main function for testing."""
    handler = LLMHandler()
    
    # Test query
    query = "How do I read a CSV file in Python?"
    response, sources = handler.query(query)
    handler.display_response(response, sources)


if __name__ == "__main__":
    main()

            