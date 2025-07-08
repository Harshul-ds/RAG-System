from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def create_vector_store_from_chunks(chunks):
    """
    Creates and persists a Chroma vector store from the given document chunks.
    This should be run only once when setting up the database.
    """
    print("Creating and persisting vector store...")
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"Vector store created successfully in '{PERSIST_DIRECTORY}'.")
    return vector_store

def get_hybrid_retriever(chunks):
    """
    Initializes and returns a hybrid retriever.
    It loads the persisted Chroma vector store and creates an in-memory BM25 retriever.
    """
    print("Initializing hybrid retriever...")

    # 1. Initialize the BM25 retriever from the document chunks (in-memory)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5  # Retrieve top 5 results
    print("  - BM25 retriever initialized.")

    # 2. Load the persisted Chroma vector store
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("  - Vector store retriever loaded from disk.")

    # 3. Create the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # Can be tuned for better performance
    )
    print("Hybrid ensemble retriever created successfully.")

    return ensemble_retriever
