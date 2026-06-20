import json
import os

from dotenv import load_dotenv
from datasets import Dataset

from rag_eval_adapter import query_rag

from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness

from ragas.llms import llm_factory
from openai import OpenAI

load_dotenv()

# ---------------------------
# OpenRouter Client
# ---------------------------

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

ragas_llm = llm_factory(
    model="openai/gpt-oss-120b:free",
    client=client,
)

print("LLM TYPE:", type(ragas_llm))

# ---------------------------
# Build Evaluation Dataset
# ---------------------------

evaluation_rows = []

with open("benchmark_dataset.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

for sample in benchmark:
    question = sample["question"]

    rag_answer, retrieved_contexts, _ = query_rag(question)

    evaluation_rows.append(
        {
            "user_input": question,
            "response": rag_answer,
            "retrieved_contexts": retrieved_contexts,
            "reference": sample["reference_answer"],
        }
    )

eval_dataset = Dataset.from_list(evaluation_rows)

print(f"Dataset Size: {len(eval_dataset)}")

# ---------------------------
# Metrics
# ---------------------------

print("Metric Type:", type(faithfulness))

# ---------------------------
# Evaluate
# ---------------------------

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness],
)
print("\n===== RESULTS =====")
print(results)