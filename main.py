# main.py - Humanized, All-in-One Version (Corrected for Distance Metric)

import logging
import os
from dotenv import load_dotenv

from pydantic_settings import BaseSettings
from langchain_core.exceptions import LangChainException
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. Centralized Configuration (with corrected threshold) ---
class Settings(BaseSettings):
    """Manages application configuration."""
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "llama3-8b-8192"
    LLM_TEMPERATURE: float = 0.7
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CACHE_COLLECTION_NAME: str = "semantic_cache_groq"
    CACHE_PERSIST_DIRECTORY: str = "./db_cache_groq"
    
    # --- FIX #1: Renamed for clarity and set a reasonable value for distance ---
    DISTANCE_THRESHOLD: float = 0.4 

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# --- 2. The Core Chatbot Logic ---
class SemanticCacheChatbot:
    """A chatbot that uses a semantic cache based on vector distance."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = None
        self.embedding_model = None
        self.vector_store = None
        
        try:
            logging.info(f"Initializing LLM: {self.settings.LLM_MODEL_NAME}")
            self.llm = ChatGroq(
                temperature=self.settings.LLM_TEMPERATURE,
                model_name=self.settings.LLM_MODEL_NAME,
                api_key=self.settings.GROQ_API_KEY
            )

            logging.info(f"Initializing embedding model: {self.settings.EMBEDDING_MODEL_NAME}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL_NAME
            )

            logging.info(f"Initializing vector store from: {self.settings.CACHE_PERSIST_DIRECTORY}")
            self.vector_store = Chroma(
                collection_name=self.settings.CACHE_COLLECTION_NAME,
                embedding_function=self.embedding_model,
                persist_directory=self.settings.CACHE_PERSIST_DIRECTORY
            )
            logging.info("Chatbot initialization complete.")
        except Exception as e:
            logging.critical(f"Failed to initialize chatbot components: {e}", exc_info=True)
            raise

    def get_response(self, user_query: str) -> str:
        """Processes a user query by checking the cache for low-distance vectors."""
        logging.info(f"Searching cache for query: '{user_query}'")
        try:
            cached_results = self.vector_store.similarity_search_with_score(query=user_query, k=1)
        except Exception as e:
            logging.error(f"Error during cache search: {e}", exc_info=True)
            return "Sorry, I'm having trouble accessing my memory right now."

        # --- FIX #2: The core logic correction ---
        # We check if a result exists AND if its distance is LESS than our threshold.
        if cached_results and cached_results[0][1] <= self.settings.DISTANCE_THRESHOLD:
            most_similar_doc, score = cached_results[0]
            cached_answer = most_similar_doc.metadata.get("answer", "Error: Cached answer not found.")
            
            logging.info(f"✅ Cache HIT! Distance: {score:.4f} <= {self.settings.DISTANCE_THRESHOLD}")
            logging.info(f"   (Found similar question: '{most_similar_doc.page_content}')")
            return f"[FROM CACHE] {cached_answer}"

        # Handle Cache Miss
        if cached_results:
            # Log why it was a miss if a result was found but was too far away
            logging.info(f"❌ Cache MISS. Closest match distance ({cached_results[0][1]:.4f}) > threshold ({self.settings.DISTANCE_THRESHOLD}).")
        else:
            logging.info("❌ Cache MISS. No similar documents found.")
        
        logging.info("Querying LLM for a new answer.")
        try:
            llm_response = self.llm.invoke(user_query)
            new_answer = llm_response.content
        except LangChainException as e:
            logging.error(f"Error querying LLM: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while trying to generate a new response."
        
        # Update the cache
        logging.info("Updating cache with new Q&A pair.")
        try:
            self.vector_store.add_texts(
                texts=[user_query], metadatas=[{"answer": new_answer}]
            )
            self.vector_store.persist()
            logging.info("Cache updated and persisted.")
        except Exception as e:
            logging.error(f"Failed to update cache: {e}", exc_info=True)
        
        return f"[FROM LLM] {new_answer}"

# --- 3. Application Entrypoint ---
def main():
    """Main function to initialize and run the chatbot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.info("--- Starting Semantic Cache Chatbot ---")
    try:
        load_dotenv()
        settings = Settings()
        chatbot = SemanticCacheChatbot(settings=settings)
    except Exception as e:
        logging.critical(f"Application failed to start: {e}", exc_info=True)
        return
    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                logging.info("--- Exiting chatbot ---")
                break
            if not user_input:
                continue
            response = chatbot.get_response(user_input)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\n")
            logging.info("--- Exiting chatbot due to user interrupt ---")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)

if __name__ == "__main__":
    main()