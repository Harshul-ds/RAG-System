from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

def create_evaluation_dataset(questions, ground_truths):
    """Creates a Hugging Face dataset for evaluation."""
    data = {"question": questions, "ground_truth": ground_truths}
    dataset = Dataset.from_dict(data)
    return dataset

def run_evaluation(dataset, qa_chain):
    """Runs the RAG evaluation and returns the results."""
    results = []
    for entry in dataset:
        response = qa_chain({"query": entry['question']})
        results.append({
            "question": entry['question'],
            "answer": response['result'],
            "contexts": [doc.page_content for doc in response['source_documents']],
            "ground_truth": entry['ground_truth']
        })

    eval_dataset = Dataset.from_list(results)

    # Evaluate the results using ragas
    score = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )

    return score

if __name__ == '__main__':
    # This is an example of how to use the evaluation framework.
    # It requires a running QA chain and a predefined set of questions and answers.
    print("Evaluation script is ready. It should be called from main.py or a notebook.")
    # from src.vector_store import load_vector_store
    # from src.retrieval_generation import create_qa_chain
    # 
    # # 1. Define your test questions and ground truth answers
    # questions = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?"]
    # ground_truths = ["Paris", "Harper Lee"]
    # 
    # # 2. Create the evaluation dataset
    # eval_dataset = create_evaluation_dataset(questions, ground_truths)
    # 
    # # 3. Load your QA chain
    # db = load_vector_store()
    # qa_chain = create_qa_chain(db)
    # 
    # # 4. Run evaluation
    # evaluation_results = run_evaluation(eval_dataset, qa_chain)
    # print(evaluation_results)
