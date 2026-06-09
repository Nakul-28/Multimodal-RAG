from backend.rag_engine import (
    get_vectorstore,
    generate_final_answer,
    rerank_chunks,
)

RERANK_CANDIDATE_K = 20


def query_rag(question, k=5):

    vs = get_vectorstore()

    try:
        print(
            "Collection Count:",
            vs._collection.count()
        )
    except Exception as e:
        print("Could not read collection count:", e)

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": max(k, RERANK_CANDIDATE_K),
            "fetch_k": max(k * 4, RERANK_CANDIDATE_K),
            "lambda_mult": 0.75,
        },
    )

    candidate_chunks = retriever.invoke(question)
    chunks = rerank_chunks(question, candidate_chunks, k)

    answer, _ = generate_final_answer(
        chunks,
        question,
        include_images=True
    )

    contexts = [
        chunk.page_content
        for chunk in chunks
    ]

    return answer, contexts, chunks
