from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)

from rag_eval_adapter import query_rag
from openrouter_judge import OpenRouterJudge

# -------------------------------
# Judge Model
# -------------------------------
judge = OpenRouterJudge(
    model_name="openai/gpt-oss-120b:free"
)

faithfulness = FaithfulnessMetric(model=judge)
relevancy = AnswerRelevancyMetric(model=judge)


# -------------------------------
# Benchmark Questions
# -------------------------------
questions = [
    "What is the main contribution of the paper?",
    "What is self-attention?",
    "What is multi-head attention?",
    "Why does the Transformer use positional encoding?",
    "What datasets were used for evaluation?",
    "What BLEU score was achieved on English-to-German translation?",
    "What are the components of the encoder?",
    "What are the components of the decoder?",
    "Why is attention divided by sqrt(dk)?",
    "How does the Transformer compare to recurrent networks?"
]
"""
    "What problem does the Transformer architecture solve?",
    
    "What is self-attention?",
    
    "Why is self-attention preferred over recurrent networks?",
    
    "What are the main components of the Transformer encoder?",
    
    "What are the main components of the Transformer decoder?",
    
    "What is multi-head attention?",
    
    "Why does the Transformer use positional encoding?",
    
    "How are positional encodings computed?",
    
    "What is scaled dot-product attention?",
    
    "Why is the attention score divided by the square root of dk?",
    
    "What are the advantages of multi-head attention over single-head attention?",
    
    "What datasets were used for evaluation?",
    
    "What BLEU score was achieved on the WMT 2014 English-to-German task?",
    
    "How does the Transformer compare to previous state-of-the-art models?",
    
    "What is the purpose of residual connections in the Transformer?",
    
    "What role does layer normalization play in the architecture?",
    
    "How many layers are used in the base Transformer model?",
    
    "What is the computational complexity of self-attention compared to recurrent layers?",
    
    "Summarize the experimental results presented in the paper."
]
"""
# -------------------------------
# Evaluation Loop
# -------------------------------
faithfulness_scores = []
relevancy_scores = []

for question in questions:

    answer, contexts, chunks = query_rag(question)

    print("\n" + "=" * 80)
    print("QUESTION:", question)

    print(f"Retrieved Chunks: {len(chunks)}")

    if chunks:
        print("\n===== FIRST CHUNK PAGE CONTENT =====")
        for i, chunk in enumerate(chunks):
            print(f"\n===== CHUNK {i+1} =====")
            print(chunk.page_content[:1000])

        print("\n===== FIRST CHUNK METADATA =====")
        print(chunks[0].metadata)
    else:
        print("No chunks retrieved.")

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
    )

    try:
        faith_score = faithfulness.measure(test_case)
        rel_score = relevancy.measure(test_case)

        faithfulness_scores.append(faith_score)
        relevancy_scores.append(rel_score)

        print("\nFaithfulness:", faith_score)
        print("Relevancy:", rel_score)

    except Exception as e:
        print(f"\nEvaluation Error: {e}")

# -------------------------------
# Summary
# -------------------------------
print("\n" + "=" * 80)
print("FINAL RESULTS")

if faithfulness_scores:
    print(
        "Average Faithfulness:",
        sum(faithfulness_scores) / len(faithfulness_scores)
    )

if relevancy_scores:
    print(
        "Average Relevancy:",
        sum(relevancy_scores) / len(relevancy_scores)
    )