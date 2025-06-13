# Your AI assistant using Langchain

AI Assistant for Website Content: A Retrieval-Augmented Generation (RAG) Solution
This project showcases how to quickly create an AI assistant by extracting content from a website, processing it, and building a Retrieval-Augmented Generation (RAG) system using Google Gemini, crawl4ai, LangChain, and ChromaDB. The example focuses on indexing content from thomsonreuters.in to create a domain-specific chatbot.

Project Purpose
The main goal of this application is to create an intelligent assistant that can answer questions about Thomson Reuters India's services, products, and general information by utilizing the official website content as the knowledge base. This ensures that the information provided is up-to-date and directly sourced from the company.

# Features
Web Crawling: Utilizes crawl4ai to scrape content from specified URLs (intended for the Thomson Reuters India website).
Data Chunking: Splits the crawled text into manageable chunks for processing.
Embedding Generation: Uses the GoogleGenerativeAIEmbeddings model to create vector representations of the text chunks.
Vector Database: Employs ChromaDB to store and index the text chunks and their embeddings, enabling efficient retrieval of relevant information.
Retrieval Augmented Generation (RAG): Combines the retrieval of relevant documents from the vector database with a Large Language Model (ChatGoogleGenerativeAI) to generate informed responses to user queries.
Persistent Storage: ChromaDB is configured to persist the crawled data to disk, avoiding the need to recrawl every time the application is run.
Setup

## Before running the code, ensure you have the following:

Python 3.8+, 
Google Cloud Project & Gemini API Key: You need a Google Cloud project with the Gemini API enabled. Obtain your GOOGLE_API_KEY from the Google AI Studio or Google Cloud Console.

#### Environment Variables
Create a .env file in the root directory of your project with the following content:


#### Install Dependencies
It's recommended to create a virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

Then install the required packages.


Note: xmltodict is typically used for XML parsing, but the code uses xml.etree.ElementTree directly. Including it is safer if other XML operations are intended. tiktoken is usually for OpenAI token counting but might be a dependency for langchain-google-genai or other parts.

# How It Works
Crawl: The crawl4ai library navigates the target website (yoursite.com via its sitemap) and extracts the main content, converting it to Markdown.
Chunking: The extracted Markdown is split into smaller, manageable chunks. The chunk_text function is designed to intelligently break text, avoiding splitting in the middle of code blocks or sentences where possible.
Enrichment: For each chunk, the Google Gemini LLM is used to generate a relevant title and a concise summary, adding valuable metadata.
Embedding: The chunk content is converted into a high-dimensional vector (embedding) using Google's embedding-001 model.
Storage: The chunk content, its metadata (including title and summary), and its embedding are stored in ChromaDB, a local vector database, enabling efficient semantic search.
RAG Query: When a user asks a question, LangChain uses the ChromaDB retriever to find the most semantically similar chunks from the indexed website content.
Generation: These retrieved chunks, along with the user's question, are fed into the Google Gemini LLM as context. The LLM then generates a comprehensive answer based only on the provided context, ensuring factual accuracy rooted in the website's content.

# Usage
The process involves two main steps: first, crawling and indexing the website data, and then, loading the indexed data to perform RAG queries.

Step 1: Crawl and Index Website Content
This step populates your chroma_db directory with the website's content embeddings.

Ensure your .env file is set up correctly.

Run the crawling part of the code. In a Colab environment, this would typically involve running the cells from import os down to await main().

Step 2: Load Indexed Data and Query
After the crawling process is complete and your chroma_db directory is populated, you can load the data and start querying.

Ensure chroma_db exists from Step 1. and load the ChromaDB instance.


#### Define the directory where the database is saved
persist_directory = "./chroma_db"

#### Re-initialize the embeddings model if it's not in your current session scope
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#### Load the ChromaDB from the persistent directory
chroma_client = Chroma(
    collection_name="crawled_data",
    embedding_function=embeddings_model, # Use the same embedding function
    persist_directory=persist_directory
)

print(f"ChromaDB loaded from {persist_directory}")

Set up the RAG chain and query:

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#### Ensure LLM and embeddings_model are defined as in the crawling step
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_output_tokens=1500)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


#### Define the prompt template for the RAG model
template = """ Define the system prompt and set the tone of the LLM"""

#### Create a retriever from the ChromaDB collection
retriever = chroma_client.as_retriever()

#### Create the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
)

# Example usage:
question = "What is Thomson Reuters India about?"
response = rag_chain.invoke(question)
print(response)

### You can now ask other questions, for example:
question = "What services does Thomson Reuters India offer for legal professionals?"

# Customization

Target Website: Modify the sitemap_url in the get_site_ai_docs_urls function (or your renamed version) to crawl a different website.
LLM & Embeddings: Easily switch to other Google Gemini models or even other LLM providers (e.g., OpenAI, Anthropic) by updating the LLM and embeddings_model initializations, provided LangChain supports them.
Prompt Engineering: Adjust the template variable in the RAG chain to refine the AI assistant's persona, instructions, and response style. This is crucial for tailoring the assistant to your specific needs.
Chunking Strategy: Modify the chunk_text function to experiment with different chunk_size values or chunking logic to optimize for your specific content type.
Vector Database: While this example uses ChromaDB, LangChain supports various other vector stores (e.g., Pinecone, Weaviate, Qdrant). You could integrate a different one if preferred.



