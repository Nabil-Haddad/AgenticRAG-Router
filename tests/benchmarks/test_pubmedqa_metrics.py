from benchmarks.pubmedqa_metrics import recall_at_k, reciprocal_rank


# recall_at_k

def test_recall_at_k_hit_at_top_rank():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_recall_at_k_hit_within_k():
    assert recall_at_k(["a", "b", "c", "d", "e"], ["c"], k=5) == 1.0


def test_recall_at_k_hit_outside_k_counts_as_miss():
    assert recall_at_k(["a", "b", "c", "d", "e"], ["c"], k=2) == 0.0


def test_recall_at_k_no_hit_anywhere():
    assert recall_at_k(["a", "b", "c"], ["z"], k=3) == 0.0


def test_recall_at_k_partial_hit_with_multiple_relevant_ids():
    # only 1 of 2 relevant ids is in the top-k - a fraction, not a hard miss
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5


def test_recall_at_k_full_hit_with_multiple_relevant_ids():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_recall_at_k_empty_relevant_ids_returns_zero():
    assert recall_at_k(["a", "b", "c"], [], k=3) == 0.0


def test_recall_at_k_empty_ranked_ids_returns_zero():
    assert recall_at_k([], ["a"], k=3) == 0.0


def test_recall_at_k_k_larger_than_ranked_list():
    assert recall_at_k(["a"], ["a"], k=100) == 1.0


def test_recall_at_k_zero_k_never_hits():
    assert recall_at_k(["a", "b"], ["a"], k=0) == 0.0


# reciprocal_rank

def test_reciprocal_rank_hit_at_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_hit_at_third_position():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3


def test_reciprocal_rank_not_cut_off_by_any_k():
    # unlike recall_at_k, a hit deep in the ranking still counts
    ranked = [str(i) for i in range(20)]
    assert reciprocal_rank(ranked, ["19"]) == 1 / 20


def test_reciprocal_rank_no_hit_returns_zero():
    assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0


def test_reciprocal_rank_uses_the_earliest_of_multiple_relevant_ids():
    assert reciprocal_rank(["a", "b", "c"], ["c", "b"]) == 1 / 2


def test_reciprocal_rank_empty_relevant_ids_returns_zero():
    assert reciprocal_rank(["a", "b", "c"], []) == 0.0


def test_reciprocal_rank_empty_ranked_ids_returns_zero():
    assert reciprocal_rank([], ["a"]) == 0.0
