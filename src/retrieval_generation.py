import boto3
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain.chains import LLMChain
from flashrank import Ranker, RerankRequest
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# --- AWS Bedrock Configuration ---
BEDROCK_CLIENT = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
MODEL_ID = 'anthropic.claude-v2'  # Or another model like 'meta.llama3-70b-instruct-v1:0'

# --- Re-ranking Configuration ---
# Using a lightweight, fast, and effective cross-encoder
RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")

from langchain.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

def get_llm():
    """Initializes the primary Bedrock LLM or falls back to a local HF pipeline."""
    try:
        llm = Bedrock(
            model_id=MODEL_ID,
            client=BEDROCK_CLIENT,
            model_kwargs={'max_tokens_to_sample': 512}
        )
        # Make a test call to ensure access
        _ = llm("Hello", stop=["."])
        print("Bedrock LLM initialized successfully.")
        return llm
    except Exception as e:
        print(f"[WARN] Bedrock not available ({e}). Falling back to local GPT-2 pipeline. This is for demo only and not production quality.")
        generator = hf_pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
        )
        return HuggingFacePipeline(pipeline=generator)

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:
"""

def get_rag_chain(llm):
    """Creates an LLMChain for generating answers based on context."""
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    rag_chain = LLMChain(llm=llm, prompt=prompt)
    return rag_chain

def generate_response(query, retriever, llm, llm_chain):
    """The complete, state-of-the-art RAG pipeline."""
    # 1. Generate multiple queries for a more comprehensive search
    print(f"\n1. Generating multiple queries for: '{query}'")
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    retrieved_docs = multi_query_retriever.get_relevant_documents(query)
    print(f"   - Retrieved {len(retrieved_docs)} documents from multiple queries.")

    # 2. Compress the context to keep only relevant information
    # This is a bit of a trick. We can't use the ContextualCompressionRetriever directly
    # with the re-ranker. So we manually create the compressor and apply it.
    print("2. Compressing retrieved documents to extract relevant snippets...")
    compressor = LLMChainExtractor.from_llm(llm)
    compressed_docs = compressor.compress_documents(documents=retrieved_docs, query=query)
    print(f"   - Compressed down to {len(compressed_docs)} relevant snippets.")

    # 3. Re-rank the compressed, relevant snippets for final context selection
    print("3. Re-ranking the compressed snippets...")
    # We need to adapt the format for flashrank, which expects a list of dicts or Passage objects
    passages_for_reranking = [
        {"id": i, "text": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(compressed_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages_for_reranking)
    reranked_docs = RANKER.rerank(rerank_request)
    top_docs = reranked_docs[:3]
    print(f"   - Re-ranked and selected top {len(top_docs)} snippets.")

    # 4. Format the final context and generate the answer
    print("4. Generating final answer...")
    context_str = "\n\n---\n\n".join([doc['text'] for doc in top_docs])
    response = llm_chain.run(context=context_str, question=query)

    return {
        "result": response,
        "source_documents": top_docs,
    }

if __name__ == '__main__':
    # This is for testing purposes and requires a fully built system.
    # Run through main.py for a complete pipeline execution.
    from src.ingestion import load_documents, split_text
    from src.retriever import get_hybrid_retriever, create_vector_store_from_chunks
    import os

    print("Running retrieval & generation test...")
    # 1. Ingest documents and create chunks
    if not os.path.exists("db"):
        print("Database not found. Creating one...")
        docs = load_documents()
        chunks = split_text(docs)
        if chunks:
            create_vector_store_from_chunks(chunks)
        else:
            print("No documents found to process.")
            exit()
    else:
        docs = load_documents()
        chunks = split_text(docs)

    # 2. Initialize the retriever and the LLM chain
    retriever = get_hybrid_retriever(chunks)
    llm = get_llm()
    rag_chain = get_rag_chain(llm)

    # 3. Run a test query
    test_query = "What is the main topic of the documents?"
    result = generate_response(test_query, retriever, llm, rag_chain)

    print("\n--- Response ---")
    print(result['result'])
    print("\n--- Source Documents ---")
    for doc in result['source_documents']:
        print(f"- {doc['metadata'].get('source', 'N/A')} (Score: {doc['score']:.4f})")
