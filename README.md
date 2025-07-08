# RAG System Prototype

This project is a production-ready prototype of a Retrieval-Augmented Generation (RAG) pipeline. It's designed to ingest documents, store them in a vector database, and use a large language model (LLM) on AWS Bedrock to answer questions based on the provided context.

## Features

- **Data Ingestion**: Loads PDF documents and splits them into manageable chunks.
- **Vectorization**: Uses `sentence-transformers` to create embeddings and stores them in `ChromaDB`.
- **Retrieval & Generation**: Employs `langchain` and `boto3` to interact with AWS Bedrock for question-answering.
- **Evaluation**: Includes a framework with `ragas` to assess the performance of the RAG pipeline.
- **Dependency Management**: Uses `uv` and a `pyproject.toml` for efficient and reproducible environment setup.

## Project Structure

```
. 
├── .venv/ # Virtual environment managed by uv
├── data/ # Place your PDF documents here
├── db/ # Persistent ChromaDB vector store
├── src/ # Source code for the RAG pipeline
│ ├── ingestion.py
│ ├── vector_store.py
│ ├── retrieval_generation.py
│ ├── evaluation.py
│ └── main.py
├── notebooks/ # Jupyter notebooks for experimentation
├── tests/ # Tests for the pipeline components
├── pyproject.toml # Project dependencies and metadata
└── README.md
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- `uv` package manager (`pip install uv`)
- AWS account with access to Bedrock and configured credentials.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harshul-ds/RAG-System-Prototype.git
    cd RAG-System-Prototype
    ```

2.  **Create the virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    uv pip sync pyproject.toml
    ```

## How to Run

1.  **Add Documents**: Place your PDF files into the `data/` directory.

2.  **Configure AWS Credentials**: Make sure your AWS credentials are set up correctly. You can do this by:
    -   Creating a `~/.aws/credentials` file.
    -   Setting the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` (if applicable) environment variables.

3.  **Run the Pipeline**: Execute the main script to start the ingestion process and ask questions.
    ```bash
    python src/main.py
    ```
    The first run will create the vector store. Subsequent runs will load the existing store.

## Evaluation

To evaluate the pipeline's performance:

1.  Open `src/main.py` and set the `run_eval` flag to `True`.
2.  Modify the `questions` and `ground_truths` lists in `src/main.py` with your evaluation data.
3.  Run the main script as described above.
