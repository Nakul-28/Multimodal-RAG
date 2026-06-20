

# Importing Libraries and Loading env


import re
import json
import os
import time

from dotenv import load_dotenv
from datasets import Dataset

from rag_eval_adapter import query_rag

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    context_precision,
    context_recall,
)

from ragas.llms import llm_factory

from openai import OpenAI
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer


load_dotenv()

# Hugging Face Authentication


hf_token = os.getenv("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    print("✅ Hugging Face Token Loaded")
else:
    print("⚠️ HF_TOKEN not found")


# Configure environment variables

BENCHMARK_FILE = "rag_benchmark_50q.json"
CACHE_FILE = "rag_outputs.json"
RESULTS_FILE = "evaluation_results.json"
GEN_TIME_FILE = "generation_times.json"
JUDGE_MODEL ="nvidia/nemotron-3-ultra-550b-a55b:free"

# Ollama Judge

class JSONSafeChatOllama(ChatOllama):
    """Sanitizes LLM output to fix unescaped backslashes before RAGAS parses it."""

    def _sanitize(self, text: str) -> str:
        try:
            json.loads(text)
            return text                         # already valid, skip
        except json.JSONDecodeError:
            # Escape backslashes NOT followed by valid JSON escape characters
            fixed = re.sub(r'\\(?!["\\/bfnrtu0-9])', r'\\\\', text)
            try:
                json.loads(fixed)
                return fixed                    # fix worked
            except json.JSONDecodeError:
                return text                     # return original, let RAGAS handle it

    def invoke(self, input, config=None, **kwargs):
        result = super().invoke(input, config, **kwargs)
        if isinstance(result.content, str):
            result.content = self._sanitize(result.content)
        return result



judge_llm = JSONSafeChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    num_ctx=8192,
)

ragas_llm = LangchainLLMWrapper(judge_llm)
print("Judge Model:", judge_llm.model)

# Embedding Model(Ollama)

OllamaEmbeddings(
    model="nomic-embed-text-v2-moe:latest"
)

# Embedding Model(Hugging Face)


from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embedding_model = LangchainEmbeddingsWrapper(hf_embeddings)

# Build / Load Dataset


if os.path.exists(CACHE_FILE):

    print(f"Loading cached outputs from {CACHE_FILE}")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        evaluation_rows = json.load(f)

    if os.path.exists(GEN_TIME_FILE):
        with open(GEN_TIME_FILE, "r", encoding="utf-8") as f:
            generation_times = json.load(f)

        print(
            f"Loaded {len(generation_times)} cached generation times"
        )

else:

    print("Generating RAG outputs...")

    evaluation_rows = []
    generation_times = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    for idx, sample in enumerate(benchmark, start=1):

        question = sample["question"]

        print(f"\n========== QUESTION {idx} ==========")
        print(question)

        print(f"[{idx}/{len(benchmark)}] Processing")

        start = time.time()

        rag_answer, retrieved_contexts, _ = query_rag(question,5)

        generation_time = time.time() - start
        generation_times.append(generation_time)

        print(
            f"QUESTION {idx} COMPLETED "
            f"({generation_time:.2f} sec)"
        )

        evaluation_rows.append(
            {
                "user_input": question,
                "response": rag_answer,
                "retrieved_contexts": retrieved_contexts,
                "reference": sample["reference_answer"],
            }
        )

        # Save after every question
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                evaluation_rows,
                f,
                indent=2,
                ensure_ascii=False,
            )

        with open(GEN_TIME_FILE, "w", encoding="utf-8") as f:
            json.dump(
                generation_times,
                f,
                indent=2,
            )

    print(f"Saved outputs to {CACHE_FILE}")
    print(f"Saved generation times to {GEN_TIME_FILE}")

# Dataset


eval_dataset = Dataset.from_list(evaluation_rows)

print(f"Dataset Size: {len(eval_dataset)}")

# Metric Configuration


faithfulness.llm = ragas_llm
context_precision.llm = ragas_llm
context_recall.llm = ragas_llm

answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = embedding_model

answer_similarity.embeddings = embedding_model

answer_correctness.llm = ragas_llm
answer_correctness.embeddings = embedding_model


metrics = [
    faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    context_precision,
    context_recall,
]

# Evaluation


print("\nStarting Evaluation...\n")

eval_start = time.time()

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    batch_size=1,
)

eval_time = time.time() - eval_start

# Results


results_dict = results.to_pandas().mean(numeric_only=True).to_dict()

if "faithfulness" in results_dict:
    results_dict["hallucination_rate"] = (
        1 - results_dict["faithfulness"]
    )

if generation_times:
    results_dict["avg_generation_latency_sec"] = (
        sum(generation_times) / len(generation_times)
    )
    results_dict["total_generation_time_sec"] = (
    sum(generation_times)
    )

results_dict["evaluation_time_sec"] = eval_time

print("\n===== RESULTS =====")

for k, v in results_dict.items():
    print(f"{k}: {v}")
    


# Save Results


with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results_dict,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nSaved results to {RESULTS_FILE}")