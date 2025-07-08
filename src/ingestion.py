import os
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
        UnstructuredMarkdownLoader,
    )
except ModuleNotFoundError:
    from langchain.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
        UnstructuredMarkdownLoader,
    )
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(data_path="data/"):
    """
    Loads documents from the specified directory, supporting .pdf, .txt, .docx, and .md files.
    """
    documents = []
    print(f"Loading documents from {os.path.abspath(data_path)}...")
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        loader = None
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)

        if loader:
            print(f"  - Loading {filename}")
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"    Error loading {filename}: {e}")
        else:
            print(f"  - Skipping unsupported file: {filename}")
            
    return documents

def split_text(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == '__main__':
    # This is for testing the ingestion process independently
    # Create a dummy data directory and a pdf file for a full test
    docs = load_documents()
    chunks = split_text(docs)
    print(f"Loaded {len(docs)} documents and split them into {len(chunks)} chunks.")
