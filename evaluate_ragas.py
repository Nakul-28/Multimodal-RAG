from datasets import Dataset
from ragas import evaluate
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from rag_eval_adapter import query_rag
import os
from ragas.metrics import Faithfulness, ResponseRelevancy
from langchain_ollama import OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")
)
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

data = {
    "user_input": [],
    "response": [],
    "retrieved_contexts": [],
}

for question in questions:

    answer, contexts, chunks = query_rag(question)

    data["user_input"].append(question)
    data["response"].append(answer)
    data["retrieved_contexts"].append(contexts)

dataset = Dataset.from_dict(data)
judge_llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(judge_llm)
relevancy_metric = ResponseRelevancy(
    llm=ragas_llm,
    embeddings=embeddings
)
result = evaluate(
    dataset,
    metrics=[
        Faithfulness(llm = ragas_llm),
        relevancy_metric,
    ]
)

print(result)