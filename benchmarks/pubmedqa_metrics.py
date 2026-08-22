def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    # fraction of the relevant set found within the top-k ranked ids -
    # reduces to a plain hit/miss (0.0 or 1.0) when there's exactly one
    # relevant id, as is always true for pqa_labeled, but stays correct
    # if a query ever has more than one relevant document
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    retrieved_top_k = set(ranked_ids[:k])
    return len(retrieved_top_k & relevant_set) / len(relevant_set)


def precision_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    # fraction of the top-k ranked ids that are relevant - unlike recall_at_k,
    # this reduces to (0.0 or 1.0)/k for pqa_labeled specifically (exactly one
    # relevant id per query), so it carries no information recall_at_k doesn't
    # already have here; kept general/correct for datasets with >1 relevant id
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_top_k = set(ranked_ids[:k])
    return len(retrieved_top_k & relevant_set) / k


def reciprocal_rank(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    # 1/rank of the first relevant id anywhere in the ranking, or 0.0 if
    # none is found - unlike recall_at_k this isn't cut off at a k, since
    # MRR is conventionally computed over the full ranking
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0
