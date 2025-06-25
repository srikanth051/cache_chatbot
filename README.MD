# Semantic Cache Chatbot

This project is a Python-based conversational AI chatbot that demonstrates a powerful and efficient industry-standard technique: **semantic caching**. Instead of just answering questions, the chatbot first checks if a semantically similar question has been asked before. If so, it retrieves the answer instantly from a persistent local cache, saving time, reducing API costs, and ensuring consistent answers. If not, it falls back to a powerful Large Language Model (LLM) to generate a new answer and then dynamically updates the cache.

## Key Features

- **Dynamic Semantic Caching:** Understands the *meaning* of questions, not just the exact wording. "Explain how to make power from the sun" will match a cached answer for "How do solar panels work?".
- **Persistent Memory:** The cache is saved to disk using a vector database, so the chatbot's knowledge persists across sessions.
- **Cost & Speed Efficient:** Drastically reduces calls to the LLM API, leading to faster responses and lower operational costs.
- **Pluggable LLM Backend:** Currently configured to use the extremely fast and free-tier-friendly Groq API (with Llama 3), but can be easily swapped for any other LangChain-compatible LLM provider (like OpenAI, Anthropic, etc.).
- **Industry-Standard Practices:** Built using virtual environments, structured logging, centralized configuration, and clear, commented code.

---

## How It Works

The chatbot follows a simple yet powerful logic flow for every user query:

1.  **Receive Query:** A user asks a question.
2.  **Create Embedding:** The question is converted into a numerical vector (an embedding) that represents its semantic meaning.
3.  **Search Cache:** The chatbot searches its local vector database (ChromaDB) for existing question vectors with the smallest "distance" (i.e., highest similarity) to the new query's vector.
4.  **Evaluate Match:**
    - **Cache Hit:** If the distance to the closest stored question is **below** a set threshold (e.g., `DISTANCE_THRESHOLD = 0.4`), it's considered a match. The associated answer is retrieved instantly.
    - **Cache Miss:** If no question is found or the closest one is too far away (distance is too high), it's a miss.
5.  **Fallback to LLM:** On a cache miss, the query is sent to the Groq API to generate a fresh answer.
6.  **Update Cache:** The new question and its LLM-generated answer are converted to an embedding and stored in the vector database, making the chatbot smarter for the next query.



---

## Project Structure

```
cache_chatbot/
├── .env                  # Stores secret API keys
├── main.py               # Main application entrypoint and all logic
├── requirements.txt      # List of Python dependencies
├── db_cache_groq/        # Directory where the persistent cache is stored (auto-generated)
└── README.md             # This file
```

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- A free Groq API key

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/srikanth051/cache_chatbot
    cd cache_chatbot
    ```

2.  **Create a Python Virtual Environment:**
    (This is a critical step to keep project dependencies isolated.)
    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # On macOS/Linux:
    source venv/bin/activate
    ```
    > **Note for Windows Users:** If you get an error about script execution being disabled, open PowerShell as an Administrator and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`. Then try activating again.

3.  **Install Dependencies:**
    Install all the required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of the project directory. Add your Groq API key to this file:
    ```
    GROQ_API_KEY="gsk_YourCopiedGroqApiKeyHere"
    ```

5.  **Run the Chatbot:**
    You're all set! Start the chatbot with:
    ```bash
    python main.py
    ```
    The application will initialize, and you can start chatting with it in your terminal. To quit, type `exit` and press Enter.

---

## Technology Stack & Frameworks

This project leverages several key frameworks to achieve its functionality:

- **LangChain (`langchain-community`, `langchain-groq`, `langchain-huggingface`):** The primary orchestration framework.
  - **Purpose:** It acts as the "glue" that connects all the different components. We use it to easily interface with the Groq LLM, manage the vector store, and handle the embedding models. It simplifies the process of building complex LLM-powered applications.

- **Groq (`langchain-groq`):** The Large Language Model (LLM) provider.
  - **Purpose:** This is the "brain" of the chatbot. When there is a cache miss, Groq's API is called to generate a human-like, intelligent response. We chose Groq for its incredible inference speed.

- **Sentence Transformers (`sentence-transformers`, `langchain-huggingface`):** The embedding model framework.
  - **Purpose:** This library provides the `all-MiniLM-L6-v2` model that turns text questions into numerical vectors. This is the core technology that enables *semantic* (meaning-based) search instead of just keyword search.

- **ChromaDB (`chromadb`):** The local vector database.
  - **Purpose:** This is the chatbot's "memory". It efficiently stores the question vectors and their corresponding answers. When a new query comes in, ChromaDB performs a very fast similarity search to find the closest matching vectors. It also handles persisting the data to disk.

- **Pydantic (`pydantic-settings`):** The configuration management framework.
  - **Purpose:** Provides a robust and clean way to manage all application settings (like model names, API keys, and thresholds). It automatically loads settings from the `.env` file and validates them, preventing common errors.