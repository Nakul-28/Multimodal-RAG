import time
from backend.rag_engine import (
    get_vectorstore,
    generate_final_answer,
    rerank_chunks,
    get_bm25_retriever
)

RERANK_CANDIDATE_K = 10


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


    t0 = time.time()

    print("START RETRIEVAL")
    
    dense_chunks = retriever.invoke(question)
    bm25_retriever = get_bm25_retriever()
    bm25_chunks = []
    if bm25_retriever:
        bm25_chunks = bm25_retriever.invoke(question)
    combined = {}
    for chunk in dense_chunks:
        combined[chunk.page_content] = chunk
    for chunk in bm25_chunks:
        combined[chunk.page_content] = chunk
    candidate_chunks = list(combined.values())

    print("RETRIEVAL DONE")

    print(
        f"Retrieval took "
        f"{time.time() - t0:.2f}s"
    )

    print(
        "Candidate Chunks Retrieved:",
        len(candidate_chunks)
    )

    for i, chunk in enumerate(candidate_chunks):
        print(
        f"Candidate Chunk {i+1}: "
        f"{len(chunk.page_content)} chars"
    )
    t1 = time.time()

    chunks = rerank_chunks(
        question,
        candidate_chunks,
        k
    )
    print("RERANK DONE")
    print(
        f"Rerank took "
        f"{time.time() - t1:.2f}s"
    )

    t2 = time.time()

    answer, _ = generate_final_answer(
        chunks,
        question,
        include_images=True
    )
    print("GENERATION DONE")
    print(
        f"Generation took "
        f"{time.time() - t2:.2f}s"
    )

    contexts = [
        chunk.page_content
        for chunk in chunks
    ]

    return answer, contexts, chunks
