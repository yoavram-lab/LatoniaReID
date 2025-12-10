import torch
import numpy as np


def labels_and_scores(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates):
    """
    Extract binary labels and similarity scores for all valid query-reference pairs.
    
    Pairs from the same date are excluded. Returns parallel lists where labels[i] indicates
    whether the pair is a match (same individual) and scores[i] is the similarity score.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
    
    Returns:
        Tuple of (labels, scores) as tensors
    """
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"

    labels = []
    scores = []
    for i in range(n):
        qi_label = query_labels[i]
        qi_date = query_dates[i]
        sim_row = similarity_matrix[i]
        for j in range(m):
            if qi_date == ref_dates[j]: # ignore pairs from same date
                continue
            labels.append(qi_label == ref_labels[j])  # 1 if same individual, 0 if different
            scores.append(sim_row[j])

    return torch.tensor(labels, dtype=int), torch.tensor(scores, dtype=float)


def recall_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    """
    Compute recall@k averaged across all queries.
    
    For each query, recall@k is the fraction of relevant items (same identity, different date)
    that appear in the top-k results.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
        k: Number of top results to consider
    
    Returns:
        Mean recall@k across all queries with at least one relevant item
    """
    recalls = []
    for i in range(len(query_labels)):
        valid = [j for j in similarity_matrix[i].argsort(descending=True)
                 if query_dates[i] != ref_dates[j]]
        relevant = [j for j in range(len(ref_labels))
                    if query_labels[i] == ref_labels[j] and query_dates[i] != ref_dates[j]]
        R = len(relevant)
        if R:
            hits = sum(query_labels[i] == ref_labels[j] for j in valid[:k])
            recalls.append(hits / R)
    return float(np.mean(recalls)) if recalls else 0.0

def precision_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    """
    Compute precision@k averaged across all queries.
    
    For each query, precision@k is the fraction of the top-k results that are relevant
    (same identity, different date).
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
        k: Number of top results to consider
    
    Returns:
        Mean precision@k across all queries
    """
    precisions = []
    for i in range(len(query_labels)):
        valid = [j for j in similarity_matrix[i].argsort(descending=True)
                 if query_dates[i] != ref_dates[j]]
        k_eff = min(k, len(valid))
        if k_eff:
            hits = sum(query_labels[i] == ref_labels[j] for j in valid[:k_eff])
            precisions.append(hits / k_eff)
    return float(np.mean(precisions)) if precisions else 0.0

def top_k_accuracy(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    """
    Compute top-k accuracy: fraction of queries where the correct identity appears in top-k.
    
    Only queries with at least one matching identity in the reference set are counted.
    Pairs from the same date are excluded from consideration.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
        k: Number of top results to consider
    
    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (
        n,
        m,
    ), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    correct = 0
    total = 0

    ref_labels_set = set(ref_labels)
    for i in range(n):
        if query_labels[i] not in ref_labels_set:  # skip if no matching identity in ref
            continue
        sorted_idx = similarity_matrix[i].argsort(descending=True)  # descending order
        valid_candidates = [
            j for j in sorted_idx if query_dates[i] != ref_dates[j]
        ]  # ignore pairs from same date

        # Check if there are any valid same-ID different-date candidates
        has_valid_match = any(
            query_labels[i] == ref_labels[j] for j in valid_candidates
        )
        if not has_valid_match:  # skip if no valid different-date matches exist
            continue

        total += 1
        # Check if any of the top-k valid candidates matches the query identity
        if any(query_labels[i] == ref_labels[j] for j in valid_candidates[:k]):
            correct += 1
    return correct / total if total > 0 else 0.0


def top_k_id_accuracy(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    """
    Compute top-k ID accuracy based on unique identities rather than individual images.
    
    For each query, collects the top-k unique identity IDs (not just top-k images) and
    checks if the correct identity appears among them. This matches the logic used in
    run_prediction_ids. Pairs from the same date are excluded.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
        k: Number of unique top IDs to consider
    
    Returns:
        Top-k ID accuracy as a float between 0 and 1
    """
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (
        n,
        m,
    ), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    correct = 0
    total = 0

    ref_labels_set = set(ref_labels)
    for i in range(n):
        if query_labels[i] not in ref_labels_set:  # skip if no matching identity in ref
            continue

        sorted_idx = similarity_matrix[i].argsort(descending=True)  # descending order

        # Collect top-k unique IDs (similar to run_prediction_ids)
        unique_ids = []
        for j in sorted_idx:
            if query_dates[i] == ref_dates[j]:  # ignore pairs from same date
                continue
            candidate_id = ref_labels[j]
            if candidate_id not in unique_ids:
                unique_ids.append(candidate_id)
                if len(unique_ids) == k:
                    break

        # Check if there are any valid same-ID different-date candidates
        has_valid_match = any(
            query_labels[i] == ref_labels[j]
            for j in sorted_idx
            if query_dates[i] != ref_dates[j]
        )
        if not has_valid_match:  # skip if no valid different-date matches exist
            continue

        total += 1
        # Check if the query ID is in the top-k unique IDs
        if query_labels[i] in unique_ids:
            correct += 1

    return correct / total if total > 0 else 0.0


def micro_precision_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    """
    Compute micro-averaged precision@k across all queries.
    
    Unlike macro-averaged precision@k, this computes a single precision by pooling all
    top-k predictions across all queries. Gives equal weight to each prediction rather
    than each query.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
        k: Number of top results to consider per query
    
    Returns:
        Micro-averaged precision@k as a float between 0 and 1
    """
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    correct = 0
    total = 0

    for i in range(n):
        sorted_idx = similarity_matrix[i].argsort(descending=True)#[::-1]  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        total += min(k, len(valid_candidates))  # count only up to k candidates
        # Check if any of the top-k valid candidates matches the query identity
        correct += sum(query_labels[i] == ref_labels[j] for j in valid_candidates[:k])
    
    return correct / total if total > 0 else 0.0


def R_precision(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates):
    """
    Compute R-precision averaged across all queries.
    
    For each query, R-precision is precision@R where R is the number of relevant items
    (same identity, different date) in the reference set. This is the fraction of relevant
    items found in the top-R results.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
    
    Returns:
        Mean R-precision across queries with at least one relevant item
    """
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    precisions = []
    
    for i in range(n):
        sorted_idx = similarity_matrix[i].argsort(descending=True)#[::-1]  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        is_relevant = [query_labels[i] == ref_labels[j] for j in valid_candidates]
        R = np.sum(is_relevant) # number of relevant items
        if R > 0: # skip i if no relevant items
            r = np.sum(is_relevant[:R]) # number of relevant items in top-R
            precisions.append(r / R)
    return float(np.mean(precisions)) if precisions else 0.0

def mean_average_precision_at_R(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates):
    """
    Compute mean average precision at R (mAP@R) across all queries.
    
    For each query with R relevant items, computes average precision over the top-R results,
    which is the mean of precision values at each position where a relevant item is found.
    Then averages across all queries.
    
    Args:
        similarity_matrix: (n, m) tensor of similarity scores
        query_labels: List of n query identity labels
        ref_labels: List of m reference identity labels
        query_dates: List of n query dates
        ref_dates: List of m reference dates
    
    Returns:
        Mean average precision@R across queries with at least one relevant item
    """
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    average_precisions = []
    
    for i in range(n):
        sorted_idx = similarity_matrix[i].argsort(descending=True)#[::-1]  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        is_relevant = np.array([query_labels[i] == ref_labels[j] for j in valid_candidates], dtype=np.int32)
        R = np.sum(is_relevant) # number of relevant items
        if R > 0: # skip i if no relevant items
            cumsum_relevant = np.cumsum(is_relevant)
            precision_at_r = cumsum_relevant[:R] / (np.arange(1, R + 1))
            average_precisions.append(np.mean(precision_at_r))

    return np.mean(average_precisions) if average_precisions else 0.0