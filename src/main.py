import os
from src.ingestion import load_documents, split_text
from src.retriever import (
    create_vector_store_from_chunks,
    get_hybrid_retriever,
    PERSIST_DIRECTORY,
)
from src.retrieval_generation import get_llm
from src.agent import create_rag_agent

def bootstrap_agent():
    """Initializes data, vector store (if missing), hybrid retriever, LLM and returns an AgentExecutor."""
    # Ingest / ensure DB
    documents = load_documents()
    chunks = split_text(documents)
    if not os.path.exists(PERSIST_DIRECTORY):
        create_vector_store_from_chunks(chunks)
    retriever = get_hybrid_retriever(chunks)
    llm = get_llm()
    return create_rag_agent(retriever, llm)

def main():
    """Main function to run the advanced RAG pipeline."""
    # --- 1. Ingestion and Vector Store Creation ---
    print("Checking for existing vector store...")
    documents = load_documents()
    if not documents:
        print("No documents found in the 'data' directory. Please add some files.")
        return

    chunks = split_text(documents)

    if not os.path.exists(PERSIST_DIRECTORY):
        print("Vector store not found. Creating a new one...")
        create_vector_store_from_chunks(chunks)
    else:
        print("Existing vector store found.")

    # --- 2. Initialize the Agent ---
    retriever = get_hybrid_retriever(chunks)
    llm = get_llm()
    agent_executor = create_rag_agent(retriever, llm)
    print("\nAgent is ready. Ask your questions!")

    # --- 3. Interact with the Agent ---
    # Replace with your own query
    query = "What are the main challenges in RAG implementation?"
    print(f"\nExecuting query with agent: {query}")
    response = agent_executor.invoke({"input": query})

    print("\n--- Agent's Final Answer ---")
    print(response['output'])

if __name__ == "__main__":
    main()
