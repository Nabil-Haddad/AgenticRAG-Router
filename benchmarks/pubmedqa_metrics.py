def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    # fraction of the relevant set found in the top-k ranked ids
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    retrieved_top_k = set(ranked_ids[:k])
    return len(retrieved_top_k & relevant_set) / len(relevant_set)


def precision_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    # fraction of the top-k ranked ids that are relevant
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_top_k = set(ranked_ids[:k])
    return len(retrieved_top_k & relevant_set) / k


def reciprocal_rank(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    # 1/rank of the first relevant id, or 0.0 if none is found
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0
