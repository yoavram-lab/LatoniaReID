#!/usr/bin/env python3
"""
Script to analyze statistics from batch_predictions.json
Shows status and decision distributions and summaries.

USAGE:
    Basic usage (default input file: batch_predictions.json):
        python batch_predictions_stats.py

    Specify custom input file:
        python batch_predictions_stats.py --input batch_predictions_gold12_united.json --threshold 344
        python batch_predictions_stats.py -i my_predictions.json

    Save statistics to JSON file:
        python batch_predictions_stats.py --output stats_output.json
        python batch_predictions_stats.py -o stats.json

    Change threshold for high-score no_match detection (default: 250):
        python batch_predictions_stats.py --threshold 300
        python batch_predictions_stats.py -t 200

    Suppress console output (useful when only saving to file):
        python batch_predictions_stats.py --quiet -o stats.json
        python batch_predictions_stats.py -q

    Combined example:
        python batch_predictions_stats.py -i batch_predictions.json -o stats.json -t 300

OPTIONS:
    -i, --input FILE        Input JSON file (default: batch_predictions.json)
    -o, --output FILE       Output JSON file for statistics (optional)
    -t, --threshold NUM     Threshold for high match scores in no_match cases (default: 250)
    -q, --quiet            Suppress console output

OUTPUT:
    The script provides:
    - Total entry counts (entries with status, decision)
    - Status distribution (percentages and counts)
    - Decision distribution (percentages and counts)
    - Cross-tabulations (Status by Decision and Decision by Status)
    - Alert for no_match cases with high match scores above threshold
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def load_batch_predictions(filepath="batch_predictions_1th.json"):
    """Load batch predictions from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_statistics(predictions):
    """Calculate statistics for status and decisions."""

    status_counts = Counter()
    decision_counts = Counter()
    status_by_decision = defaultdict(lambda: Counter())
    decision_by_status = defaultdict(lambda: Counter())

    total_entries = 0
    entries_with_status = 0
    entries_with_decision = 0

    # Collect statistics
    for key, entry in predictions.items():
        total_entries += 1

        status = entry.get("status")
        decision = entry.get("decision")

        if status:
            entries_with_status += 1
            status_counts[status] += 1

        if decision:
            entries_with_decision += 1
            decision_counts[decision] += 1

        # Cross-tabulation
        if status and decision:
            status_by_decision[decision][status] += 1
            decision_by_status[status][decision] += 1

    return {
        "total_entries": total_entries,
        "entries_with_status": entries_with_status,
        "entries_with_decision": entries_with_decision,
        "status_counts": dict(status_counts),
        "decision_counts": dict(decision_counts),
        "status_by_decision": {k: dict(v) for k, v in status_by_decision.items()},
        "decision_by_status": {k: dict(v) for k, v in decision_by_status.items()},
    }


def check_no_match_high_scores(predictions, threshold=250):
    """Check for queries with no match but high match scores."""
    high_score_no_match = []
    proven_below_threshold = []

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Get match score from predictions array (first prediction's match_count)
        match_score = None
        best_match_registry_id = None
        best_match_image = None

        if "predictions" in entry and len(entry["predictions"]) > 0:
            # Get the top prediction (rank 1)
            top_prediction = entry["predictions"][0]
            match_score = top_prediction.get("match_count")
            best_match_registry_id = top_prediction.get("registry_id")
            best_match_image = top_prediction.get("best_match_image")

        # Check if decision is 'no_match' and match_score > threshold
        if (
            decision == "no_match"
            and match_score is not None
            and match_score > threshold
        ):
            high_score_no_match.append(
                {
                    "individual_id": key,
                    "match_score": match_score,
                    "status": status,
                    "best_query_image": entry.get("best_query_image"),
                    "best_match_image": best_match_image,
                    "best_match_registry_id": best_match_registry_id,
                    "query_count": len(entry.get("query_images", [])),
                }
            )

        # Check if status is 'proven' and match_score < threshold
        if (
            status == "confirmed"
            and match_score is not None
            and match_score < threshold
        ):
            proven_below_threshold.append(
                {
                    "individual_id": key,
                    "match_score": match_score,
                    "decision": decision,
                    "best_query_image": entry.get("best_query_image"),
                    "best_match_image": best_match_image,
                    "best_match_registry_id": best_match_registry_id,
                    "query_count": len(entry.get("query_images", [])),
                }
            )

    return high_score_no_match, proven_below_threshold


def compute_score_statistics(predictions):
    """Compute score statistics for no_match and confirmed entries."""
    no_match_scores = []
    confirmed_scores = []

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Get match score from top prediction
        if "predictions" in entry and len(entry["predictions"]) > 0:
            top_prediction = entry["predictions"][0]
            match_score = top_prediction.get("match_count")

            if match_score is not None:
                # Collect scores for no_match decisions
                if decision == "no_match":
                    no_match_scores.append(match_score)

                # Collect scores for confirmed entries with any match (not no_match)
                if status == "confirmed" and decision and decision != "no_match":
                    confirmed_scores.append(match_score)

    return no_match_scores, confirmed_scores


def print_score_statistics(no_match_scores, confirmed_scores):
    """Print score statistics for no_match and confirmed entries."""
    print("=" * 70)
    print("MATCH SCORE STATISTICS")
    print("=" * 70)
    print()

    # No match statistics
    print("NO_MATCH entries (top-1 prediction scores):")
    if no_match_scores:
        print(f"  Count: {len(no_match_scores)}")
        print(f"  Mean:  {sum(no_match_scores) / len(no_match_scores):.1f}")
        print(f"  Min:   {min(no_match_scores)}")
        print(f"  Max:   {max(no_match_scores)}")
    else:
        print("  No entries found")
    print()

    # Confirmed match statistics
    print("CONFIRMED entries with matches (top-1 prediction scores):")
    if confirmed_scores:
        print(f"  Count: {len(confirmed_scores)}")
        print(f"  Mean:  {sum(confirmed_scores) / len(confirmed_scores):.1f}")
        print(f"  Min:   {min(confirmed_scores)}")
        print(f"  Max:   {max(confirmed_scores)}")
    else:
        print("  No entries found")
    print()


def parse_decision_ranks(decision):
    """Parse decision string to get set of correct ranks.

    E.g., "top1,top2" -> {1, 2}, "no_match" -> set()
    """
    if not decision or decision == "no_match":
        return set()

    correct_ranks = set()
    for part in decision.split(","):
        part = part.strip()
        if part.startswith("top"):
            try:
                rank = int(part[3:])
                correct_ranks.add(rank)
            except ValueError:
                pass
    return correct_ranks


def compute_threshold_precision_recall(predictions, threshold):
    """
    Compute precision and recall based on threshold.

    Definition:
    - For each prediction in an entry: score >= threshold -> predict as MATCH
    - Ground truth: decision field specifies which ranks are correct (e.g., "top1,top2")
    - Each correct match is counted separately as TP or FN
    - Each incorrect prediction above threshold is FP

    Returns: (tp, fp, tn, fn, precision, recall)
    """
    tp = 0  # Predicted match (score >= threshold), actual match (rank in decision)
    fp = 0  # Predicted match (score >= threshold), NOT actual match
    tn = 0  # Predicted no match (score < threshold), NOT actual match
    fn = 0  # Predicted no match (score < threshold), actual match (missed)

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Skip entries without decision or not confirmed
        if not decision or status != "confirmed":
            continue

        # Parse which ranks are correct matches
        correct_ranks = parse_decision_ranks(decision)

        # Iterate through all predictions
        if "predictions" not in entry:
            continue

        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if rank is None or match_score is None:
                continue

            predicted_match = match_score >= threshold
            actual_match = rank in correct_ranks

            # Update confusion matrix
            if predicted_match and actual_match:
                tp += 1
            elif predicted_match and not actual_match:
                fp += 1
            elif not predicted_match and not actual_match:
                tn += 1
            else:  # not predicted_match and actual_match
                fn += 1

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return tp, fp, tn, fn, precision, recall


def collect_fp_cases(predictions, threshold):
    """Collect FP cases: score >= threshold but not confirmed."""
    fp_cases = []

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        if not decision or status != "confirmed":
            continue

        correct_ranks = parse_decision_ranks(decision)

        if "predictions" not in entry:
            continue

        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if rank is None or match_score is None:
                continue

            # FP: score >= threshold but rank not confirmed
            if match_score >= threshold and rank not in correct_ranks:
                fp_cases.append(
                    {
                        "individual_id": key,
                        "rank": rank,
                        "score": match_score,
                        "decision": decision,
                        "registry_id": pred.get("registry_id"),
                        "best_match_image": pred.get("best_match_image"),
                    }
                )

    return fp_cases


def print_fp_cases(fp_cases, threshold):
    """Print FP cases."""
    print("=" * 70)
    print(f"FALSE POSITIVES (score >= {threshold}, not confirmed)")
    print("=" * 70)
    print()

    if not fp_cases:
        print("No FP cases found.")
        print()
        return

    print(f"Found {len(fp_cases)} FP cases:")
    print()

    for case in sorted(fp_cases, key=lambda x: x["score"], reverse=True):
        print(
            f"  {case['individual_id']} | top{case['rank']} | score={case['score']} | decision={case['decision']}"
        )
        print(f"    registry_id: {case['registry_id']}")
        if case["best_match_image"]:
            print(f"    best_match_image: {case['best_match_image']}")
        print()


def collect_fn_cases(predictions, threshold):
    """Collect FN cases: score < threshold but confirmed as match."""
    fn_cases = []

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        if not decision or status != "confirmed":
            continue

        correct_ranks = parse_decision_ranks(decision)

        if "predictions" not in entry:
            continue

        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if rank is None or match_score is None:
                continue

            # FN: score < threshold but rank IS confirmed
            if match_score < threshold and rank in correct_ranks:
                fn_cases.append(
                    {
                        "individual_id": key,
                        "rank": rank,
                        "score": match_score,
                        "decision": decision,
                        "registry_id": pred.get("registry_id"),
                        "best_match_image": pred.get("best_match_image"),
                    }
                )

    return fn_cases


def print_fn_cases(fn_cases, threshold):
    """Print FN cases."""
    print("=" * 70)
    print(f"FALSE NEGATIVES (score < {threshold}, but confirmed)")
    print("=" * 70)
    print()

    if not fn_cases:
        print("No FN cases found.")
        print()
        return

    print(f"Found {len(fn_cases)} FN cases:")
    print()

    for case in sorted(fn_cases, key=lambda x: x["score"], reverse=True):
        print(
            f"  {case['individual_id']} | top{case['rank']} | score={case['score']} | decision={case['decision']}"
        )
        print(f"    registry_id: {case['registry_id']}")
        if case["best_match_image"]:
            print(f"    best_match_image: {case['best_match_image']}")
        print()


def analyze_outcome_cases(predictions, threshold):
    """
    Analyze each query based on Top-1 prediction status and score.

    Cases:
    1. Top-1 Confirmed AND score >= threshold
    2. Top-1 Confirmed AND score < threshold
    3. Top-1 Rejected (not confirmed) AND score > threshold
    4. Top-1 Rejected AND score <= threshold
    """
    case_1 = []  # Top-1 confirmed, score >= t
    case_2 = []  # Top-1 confirmed, score < t
    case_3 = []  # Top-1 rejected, score > t
    case_4 = []  # Top-1 rejected, score <= t
    high_score_not_confirmed = []  # Any prediction with score > t that's not confirmed

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Skip entries without decision or not confirmed
        if not decision or status != "confirmed":
            continue

        # Parse which ranks are correct matches
        correct_ranks = parse_decision_ranks(decision)

        if "predictions" not in entry or len(entry["predictions"]) == 0:
            continue

        # Get top-1 prediction info
        top1_pred = entry["predictions"][0]
        top1_score = top1_pred.get("match_count")
        top1_confirmed = 1 in correct_ranks

        if top1_score is None:
            continue

        # Categorize based on top-1 status and score
        case_info = {
            "individual_id": key,
            "decision": decision,
            "top1_score": top1_score,
        }

        if top1_confirmed and top1_score >= threshold:
            case_1.append(case_info)
        elif top1_confirmed and top1_score < threshold:
            case_2.append(case_info)
        elif not top1_confirmed and top1_score > threshold:
            case_3.append(case_info)
        else:  # not top1_confirmed and top1_score <= threshold
            case_4.append(case_info)

        # Check for ANY prediction with score >= threshold that's not confirmed
        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if (
                match_score is not None
                and match_score >= threshold
                and rank not in correct_ranks
            ):
                high_score_not_confirmed.append(
                    {
                        "individual_id": key,
                        "rank": rank,
                        "score": match_score,
                        "decision": decision,
                        "registry_id": pred.get("registry_id"),
                    }
                )

    return case_1, case_2, case_3, case_4, high_score_not_confirmed


def compare_top1_vs_threshold_methods(predictions, threshold):
    """
    Compare Top-1 method vs Threshold method comprehensively.

    Top-1 method: Accept if rank 1 is confirmed, reject otherwise
    Threshold method: Accept if ANY prediction >= threshold is confirmed,
                     reject if no predictions >= threshold OR all predictions >= threshold are rejected
    """
    both_accept = []  # Both methods accept
    both_reject = []  # Both methods reject
    top1_accept_threshold_reject = []  # Top-1 accepts, Threshold rejects
    top1_reject_threshold_accept = []  # Top-1 rejects, Threshold accepts

    # Additional categories for threshold method details
    no_predictions_above_threshold = []  # No predictions >= t
    has_predictions_all_rejected = []  # Has predictions >= t but all rejected
    has_predictions_some_confirmed = (
        []
    )  # Has predictions >= t with at least 1 confirmed

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        if not decision or status != "confirmed":
            continue

        correct_ranks = parse_decision_ranks(decision)

        if "predictions" not in entry or len(entry["predictions"]) == 0:
            continue

        # Top-1 method decision
        top1_confirmed = 1 in correct_ranks
        top1_accepts = top1_confirmed

        # Threshold method analysis
        predictions_above_threshold = []
        confirmed_above_threshold = []
        rejected_above_threshold = []

        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if match_score is not None and match_score >= threshold:
                predictions_above_threshold.append(rank)
                if rank in correct_ranks:
                    confirmed_above_threshold.append(rank)
                else:
                    rejected_above_threshold.append(rank)

        threshold_accepts = len(confirmed_above_threshold) > 0

        # Categorize
        query_info = {
            "individual_id": key,
            "decision": decision,
            "top1_score": (
                entry["predictions"][0].get("match_count")
                if entry["predictions"]
                else None
            ),
            "num_predictions_above_t": len(predictions_above_threshold),
            "num_confirmed_above_t": len(confirmed_above_threshold),
            "num_rejected_above_t": len(rejected_above_threshold),
        }

        if top1_accepts and threshold_accepts:
            both_accept.append(query_info)
        elif not top1_accepts and not threshold_accepts:
            both_reject.append(query_info)
        elif top1_accepts and not threshold_accepts:
            top1_accept_threshold_reject.append(query_info)
        else:  # not top1_accepts and threshold_accepts
            top1_reject_threshold_accept.append(query_info)

        # Categorize by threshold method details
        if len(predictions_above_threshold) == 0:
            no_predictions_above_threshold.append(query_info)
        elif len(confirmed_above_threshold) == 0:
            has_predictions_all_rejected.append(query_info)
        else:
            has_predictions_some_confirmed.append(query_info)

    return (
        both_accept,
        both_reject,
        top1_accept_threshold_reject,
        top1_reject_threshold_accept,
        no_predictions_above_threshold,
        has_predictions_all_rejected,
        has_predictions_some_confirmed,
    )


def print_method_comparison(
    both_accept,
    both_reject,
    top1_accept_thresh_reject,
    top1_reject_thresh_accept,
    no_pred_above_t,
    all_rejected_above_t,
    some_confirmed_above_t,
    threshold,
):
    """Print comparison between Top-1 and Threshold methods."""
    print("=" * 70)
    print(f"METHOD COMPARISON: Top-1 vs Threshold (t={threshold})")
    print("=" * 70)
    print()

    total = (
        len(both_accept)
        + len(both_reject)
        + len(top1_accept_thresh_reject)
        + len(top1_reject_thresh_accept)
    )

    # Summary counts
    top1_accepts = len(both_accept) + len(top1_accept_thresh_reject)
    top1_rejects = len(both_reject) + len(top1_reject_thresh_accept)
    threshold_accepts = len(both_accept) + len(top1_reject_thresh_accept)
    threshold_rejects = len(both_reject) + len(top1_accept_thresh_reject)

    print("Method Performance:")
    print(f"  Top-1 method:      Accepts {top1_accepts:3d} | Rejects {top1_rejects:3d}")
    print(
        f"  Threshold method:  Accepts {threshold_accepts:3d} | Rejects {threshold_rejects:3d}"
    )
    print()

    # Threshold method breakdown
    print("Threshold Method Breakdown:")
    print(
        f"  No predictions >= {threshold}:                    {len(no_pred_above_t):3d} (must reject)"
    )
    print(
        f"  Has predictions >= {threshold}, all rejected:     {len(all_rejected_above_t):3d} (reject)"
    )
    print(
        f"  Has predictions >= {threshold}, ≥1 confirmed:     {len(some_confirmed_above_t):3d} (accept)"
    )
    print(
        f"  Total: {len(no_pred_above_t) + len(all_rejected_above_t) + len(some_confirmed_above_t)}"
    )
    print()

    # Agreement/Disagreement table
    print("Agreement Analysis:")
    print("┌────────────────────────┬───────────────────────┬───────┬─────────┐")
    print("│ Top-1 Method           │ Threshold Method      │ Count │ Percent │")
    print("├────────────────────────┼───────────────────────┼───────┼─────────┤")
    print(
        f"│ Accept                 │ Accept                │ {len(both_accept):5d} │ {len(both_accept)/total*100:6.1f}% │"
    )
    print(
        f"│ Reject                 │ Reject                │ {len(both_reject):5d} │ {len(both_reject)/total*100:6.1f}% │"
    )
    print(
        f"│ Accept                 │ Reject                │ {len(top1_accept_thresh_reject):5d} │ {len(top1_accept_thresh_reject)/total*100:6.1f}% │"
    )
    print(
        f"│ Reject                 │ Accept                │ {len(top1_reject_thresh_accept):5d} │ {len(top1_reject_thresh_accept)/total*100:6.1f}% │"
    )
    print("└────────────────────────┴───────────────────────┴───────┴─────────┘")
    print()

    agreement = len(both_accept) + len(both_reject)
    disagreement = len(top1_accept_thresh_reject) + len(top1_reject_thresh_accept)
    print(f"Total agreement:    {agreement:3d} ({agreement/total*100:.1f}%)")
    print(f"Total disagreement: {disagreement:3d} ({disagreement/total*100:.1f}%)")
    print()

    # Detailed analysis of disagreements
    print("-" * 70)
    print("DISAGREEMENT DETAILS:")
    print("-" * 70)
    print()

    if top1_accept_thresh_reject:
        print(
            f"Top-1 ACCEPTS but Threshold REJECTS ({len(top1_accept_thresh_reject)} cases):"
        )
        print(f"  (Top-1 is correct but no confirmed predictions >= {threshold})")
        for q in sorted(
            top1_accept_thresh_reject,
            key=lambda x: x["top1_score"] if x["top1_score"] else 0,
            reverse=True,
        )[:10]:
            print(
                f"    {q['individual_id']:20s} | decision: {q['decision']:10s} | top1: {q['top1_score']:3d} | preds_≥t: {q['num_predictions_above_t']}"
            )
        if len(top1_accept_thresh_reject) > 10:
            print(f"    ... and {len(top1_accept_thresh_reject) - 10} more")
        print()

    if top1_reject_thresh_accept:
        print(
            f"Top-1 REJECTS but Threshold ACCEPTS ({len(top1_reject_thresh_accept)} cases):"
        )
        print(f"  (Top-1 is wrong but another rank >= {threshold} is correct)")
        for q in sorted(
            top1_reject_thresh_accept,
            key=lambda x: x["num_confirmed_above_t"],
            reverse=True,
        )[:10]:
            print(
                f"    {q['individual_id']:20s} | decision: {q['decision']:10s} | top1: {q['top1_score']:3d} | confirmed_≥t: {q['num_confirmed_above_t']} | rejected_≥t: {q['num_rejected_above_t']}"
            )
        if len(top1_reject_thresh_accept) > 10:
            print(f"    ... and {len(top1_reject_thresh_accept) - 10} more")
        print()

    print("=" * 70)
    print()


def print_outcome_analysis(
    case_1, case_2, case_3, case_4, high_score_not_confirmed, threshold
):
    """Print outcome analysis table."""
    print("=" * 70)
    print(f"OUTCOME ANALYSIS (Threshold = {threshold})")
    print("=" * 70)
    print()
    print("Analysis based on Top-1 prediction:")
    print()

    # Print table
    print("┌──────┬─────────────────┬──────────────────┬───────┐")
    print("│ Case │ Top-1 Status    │ Top-1 Score      │ Count │")
    print("├──────┼─────────────────┼──────────────────┼───────┤")
    print(
        f"│  1   │ Confirmed       │ score >= {int(threshold):3d}     │ {len(case_1):5d} │"
    )
    print(
        f"│  2   │ Confirmed       │ score <  {int(threshold):3d}     │ {len(case_2):5d} │"
    )
    print(
        f"│  3   │ Rejected        │ score >  {int(threshold):3d}     │ {len(case_3):5d} │"
    )
    print(
        f"│  4   │ Rejected        │ score <= {int(threshold):3d}     │ {len(case_4):5d} │"
    )
    print("└──────┴─────────────────┴──────────────────┴───────┘")
    print()

    total = len(case_1) + len(case_2) + len(case_3) + len(case_4)
    print(f"Total confirmed queries analyzed: {total}")
    print()

    # Print percentages
    if total > 0:
        print("Percentages:")
        print(f"  Case 1 (Confirmed, score >= t): {len(case_1)/total*100:5.1f}%")
        print(f"  Case 2 (Confirmed, score <  t): {len(case_2)/total*100:5.1f}%")
        print(f"  Case 3 (Rejected,  score >  t): {len(case_3)/total*100:5.1f}%")
        print(f"  Case 4 (Rejected,  score <= t): {len(case_4)/total*100:5.1f}%")
        print()

    print("-" * 70)
    print("HIGH SCORE NOT CONFIRMED CHECK:")
    print("-" * 70)

    # Check for predictions with score >= threshold that are not confirmed
    if high_score_not_confirmed:
        print(
            f"⚠️  WARNING: Found {len(high_score_not_confirmed)} predictions with score >= {threshold} that are NOT confirmed:"
        )
        print()
        for case in sorted(
            high_score_not_confirmed, key=lambda x: x["score"], reverse=True
        ):
            print(
                f"  {case['individual_id']:20s} | rank: {case['rank']} | score: {case['score']:3d} | decision: {case['decision']:10s} | registry_id: {case['registry_id']}"
            )
        print()
    else:
        print(f"✓ No predictions with score >= {threshold} that are not confirmed")
        print()

    print("=" * 70)
    print()


def print_threshold_precision_recall(tp, fp, tn, fn, precision, recall, threshold):
    """Print precision and recall based on threshold."""
    print("=" * 70)
    print(f"THRESHOLD-BASED EVALUATION (threshold={threshold})")
    print("=" * 70)
    print()
    print("Definition:")
    print("  - Positive: entry+rank confirmed in decision (e.g., top1,top2)")
    print("  - Negative: unconfirmed ranks or no_match entries")
    print()
    print(f"  Total predictions evaluated: {tp + fp + tn + fn}")
    print(f"  #TP (score >= t, confirmed):     {tp}")
    print(f"  #FP (score >= t, not confirmed): {fp}")
    print(f"  #FN (score < t, confirmed):      {fn}")
    print(f"  #TN (score < t, not confirmed):  {tn}")
    print()
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print()


def analyze_top1_statistics(predictions):
    """
    Analyze top-1 prediction statistics.

    Returns:
        top1_confirmed: Count of queries where top-1 is confirmed
        others_confirmed: Count of queries where other ranks (>1) are confirmed but not top-1
        total_queries: Total confirmed queries analyzed
    """
    top1_confirmed = 0
    others_confirmed = 0

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Skip entries without decision or not confirmed
        if not decision or status != "confirmed":
            continue

        # Skip no_match decisions
        if decision == "no_match":
            continue

        # Parse which ranks are correct
        correct_ranks = parse_decision_ranks(decision)

        if not correct_ranks:
            continue

        # Check if top-1 is confirmed
        if 1 in correct_ranks:
            top1_confirmed += 1
        else:
            others_confirmed += 1

    total_queries = top1_confirmed + others_confirmed
    return top1_confirmed, others_confirmed, total_queries


def print_top1_statistics(top1_confirmed, others_confirmed, total_queries):
    """Print top-1 prediction statistics."""
    print("=" * 70)
    print("TOP-1 PREDICTION STATISTICS")
    print("=" * 70)
    print()
    print("Analysis of query-level confirmations (excluding no_match):")
    print()
    print(f"  Total queries with matches:        {total_queries:5d}")
    print(
        f"  Top-1 confirmed:                   {top1_confirmed:5d} ({top1_confirmed/total_queries*100:5.1f}%)"
        if total_queries > 0
        else f"  Top-1 confirmed:                   {top1_confirmed:5d}"
    )
    print(
        f"  Other ranks confirmed (not top-1): {others_confirmed:5d} ({others_confirmed/total_queries*100:5.1f}%)"
        if total_queries > 0
        else f"  Other ranks confirmed (not top-1): {others_confirmed:5d}"
    )
    print()
    print("=" * 70)
    print()


def analyze_prediction_depth(predictions):
    """
    Analyze how many predictions each query has.

    Returns:
        queries_by_depth: Dict mapping number of predictions -> count of queries
        queries_without_rank3: Count of queries without rank 3 prediction
        single_pred_no_match: Count of queries with 1 prediction and no_match decision
        single_pred_total: Total queries with 1 prediction
        total_queries: Total queries analyzed
    """
    queries_by_depth = {}
    queries_without_rank3 = 0
    single_pred_no_match = 0
    single_pred_total = 0
    total_queries = 0

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Skip entries without decision or not confirmed
        if not decision or status != "confirmed":
            continue

        if "predictions" not in entry or len(entry["predictions"]) == 0:
            continue

        total_queries += 1

        # Count number of predictions
        num_predictions = len(entry["predictions"])

        if num_predictions not in queries_by_depth:
            queries_by_depth[num_predictions] = 0
        queries_by_depth[num_predictions] += 1

        # Check if query has rank 3 prediction
        has_rank3 = any(pred.get("rank") == 3 for pred in entry["predictions"])
        if not has_rank3:
            queries_without_rank3 += 1

        # Track queries with 1 prediction
        if num_predictions == 1:
            single_pred_total += 1
            if decision == "no_match":
                single_pred_no_match += 1

    return (
        queries_by_depth,
        queries_without_rank3,
        single_pred_no_match,
        single_pred_total,
        total_queries,
    )


def print_prediction_depth(
    queries_by_depth,
    queries_without_rank3,
    single_pred_no_match,
    single_pred_total,
    total_queries,
):
    """Print prediction depth statistics."""
    print("=" * 70)
    print("PREDICTION DEPTH STATISTICS")
    print("=" * 70)
    print()
    print("Distribution of queries by number of predictions:")
    print()

    for depth in sorted(queries_by_depth.keys()):
        count = queries_by_depth[depth]
        pct = count / total_queries * 100 if total_queries > 0 else 0
        print(f"  {depth:2d} prediction(s): {count:3d} queries ({pct:5.1f}%)")

    print()
    print(f"Total queries analyzed: {total_queries}")
    print()
    print(
        f"Queries WITHOUT rank 3 prediction (< 3 predictions): {queries_without_rank3} ({queries_without_rank3/total_queries*100:5.1f}%)"
        if total_queries > 0
        else f"Queries WITHOUT rank 3 prediction: {queries_without_rank3}"
    )
    print(
        f"Queries WITH rank 3 prediction (>= 3 predictions):   {total_queries - queries_without_rank3} ({(total_queries - queries_without_rank3)/total_queries*100:5.1f}%)"
        if total_queries > 0
        else f"Queries WITH rank 3 prediction: {total_queries - queries_without_rank3}"
    )
    print()

    # Print single prediction breakdown
    if single_pred_total > 0:
        print("Queries with 1 prediction breakdown:")
        print(
            f"  no_match decision: {single_pred_no_match:3d} ({single_pred_no_match/single_pred_total*100:5.1f}%)"
        )
        print(
            f"  Other decisions:   {single_pred_total - single_pred_no_match:3d} ({(single_pred_total - single_pred_no_match)/single_pred_total*100:5.1f}%)"
        )
        print()

    print("=" * 70)
    print()


def analyze_threshold_statistics(predictions, threshold):
    """
    Analyze threshold-based prediction statistics.

    Returns:
        total_above_t: Total predictions with score >= threshold
        confirmed_above_t: Confirmed predictions with score >= threshold
        rejected_above_t: Rejected predictions with score >= threshold
        confirmed_below_t: Confirmed predictions with score < threshold
    """
    total_above_t = 0
    confirmed_above_t = 0
    rejected_above_t = 0
    confirmed_below_t = 0

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Skip entries without decision or not confirmed
        if not decision or status != "confirmed":
            continue

        # Parse which ranks are correct
        correct_ranks = parse_decision_ranks(decision)

        if "predictions" not in entry:
            continue

        # Analyze each prediction
        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if rank is None or match_score is None:
                continue

            is_confirmed = rank in correct_ranks
            is_above_threshold = match_score >= threshold

            if is_above_threshold:
                total_above_t += 1
                if is_confirmed:
                    confirmed_above_t += 1
                else:
                    rejected_above_t += 1
            else:
                if is_confirmed:
                    confirmed_below_t += 1

    return total_above_t, confirmed_above_t, rejected_above_t, confirmed_below_t


def print_threshold_statistics(
    total_above_t, confirmed_above_t, rejected_above_t, confirmed_below_t, threshold
):
    """Print threshold-based prediction statistics."""
    print("=" * 70)
    print(f"THRESHOLD-BASED PREDICTION STATISTICS (t={threshold})")
    print("=" * 70)
    print()
    print("Analysis of all predictions across all queries:")
    print()
    print(f"  Total predictions >= {threshold}:     {total_above_t:5d}")
    print(
        f"    Confirmed predictions >= {threshold}: {confirmed_above_t:5d} ({confirmed_above_t/total_above_t*100:5.1f}%)"
        if total_above_t > 0
        else f"    Confirmed predictions >= {threshold}: {confirmed_above_t:5d}"
    )
    print(
        f"    Rejected predictions >= {threshold}:  {rejected_above_t:5d} ({rejected_above_t/total_above_t*100:5.1f}%)"
        if total_above_t > 0
        else f"    Rejected predictions >= {threshold}:  {rejected_above_t:5d}"
    )
    print()
    print(f"  Confirmed predictions < {threshold}:  {confirmed_below_t:5d}")
    print()
    print("=" * 70)
    print()


def analyze_queries_by_predictions_above_threshold(predictions, threshold):
    """
    Analyze queries by number of predictions above threshold.

    Returns:
        queries_by_count: Dict mapping count -> list of query info
        queries_with_multiple: List of queries with >1 prediction above threshold
    """
    queries_by_count = {}
    queries_with_multiple = []

    for key, entry in predictions.items():
        decision = entry.get("decision")
        status = entry.get("status")

        # Skip entries without decision or not confirmed
        if not decision or status != "confirmed":
            continue

        # Parse which ranks are correct
        correct_ranks = parse_decision_ranks(decision)

        if "predictions" not in entry:
            continue

        # Count predictions above threshold
        predictions_above_t = []
        confirmed_above_t = []
        rejected_above_t = []

        for pred in entry["predictions"]:
            rank = pred.get("rank")
            match_score = pred.get("match_count")

            if match_score is not None and match_score >= threshold:
                pred_info = {
                    "rank": rank,
                    "score": match_score,
                    "registry_id": pred.get("registry_id"),
                    "confirmed": rank in correct_ranks,
                }
                predictions_above_t.append(pred_info)

                if rank in correct_ranks:
                    confirmed_above_t.append(pred_info)
                else:
                    rejected_above_t.append(pred_info)

        count = len(predictions_above_t)

        if count > 0:
            query_info = {
                "query_id": key,
                "decision": decision,
                "num_above_t": count,
                "num_confirmed_above_t": len(confirmed_above_t),
                "num_rejected_above_t": len(rejected_above_t),
                "predictions_above_t": predictions_above_t,
            }

            # Add to count-based grouping
            if count not in queries_by_count:
                queries_by_count[count] = []
            queries_by_count[count].append(query_info)

            # Add to multiple predictions list
            if count > 1:
                queries_with_multiple.append(query_info)

    return queries_by_count, queries_with_multiple


def print_queries_multiple_above_threshold(
    queries_by_count, queries_with_multiple, threshold
):
    """Print queries with multiple predictions above threshold."""
    print("=" * 70)
    print(f"QUERIES BY NUMBER OF PREDICTIONS >= {threshold}")
    print("=" * 70)
    print()

    # Print distribution
    print("Distribution of queries by number of predictions >= threshold:")
    print()
    total_queries_with_any = sum(len(queries) for queries in queries_by_count.values())

    for count in sorted(queries_by_count.keys()):
        num_queries = len(queries_by_count[count])
        pct = (
            num_queries / total_queries_with_any * 100
            if total_queries_with_any > 0
            else 0
        )
        print(
            f"  {count:2d} prediction(s) >= t: {num_queries:3d} queries ({pct:5.1f}%)"
        )

    print()
    print(
        f"Total queries with any predictions >= {threshold}: {total_queries_with_any}"
    )
    print(f"Queries with >1 prediction >= {threshold}: {len(queries_with_multiple)}")
    print()

    # Print details of queries with multiple predictions
    if queries_with_multiple:
        print("-" * 70)
        print(f"QUERIES WITH MULTIPLE PREDICTIONS >= {threshold}:")
        print("-" * 70)
        print()

        # Sort by number of predictions above threshold (descending)
        for query in sorted(
            queries_with_multiple, key=lambda x: x["num_above_t"], reverse=True
        ):
            print(
                f"{query['query_id']:20s} | decision: {query['decision']:15s} | {query['num_above_t']} predictions >= {threshold}"
            )
            print(
                f"  Confirmed: {query['num_confirmed_above_t']}, Rejected: {query['num_rejected_above_t']}"
            )

            # Show each prediction above threshold
            for pred in query["predictions_above_t"]:
                status_str = "✓ confirmed" if pred["confirmed"] else "✗ rejected"
                print(
                    f"    rank {pred['rank']:2d}: score={pred['score']:3d} | {pred['registry_id']:20s} | {status_str}"
                )
            print()
    else:
        print("No queries with multiple predictions above threshold.")
        print()

    print("=" * 70)
    print()


def print_statistics(stats):
    """Print statistics in a readable format."""

    print("=" * 70)
    print("BATCH PREDICTIONS STATISTICS")
    print("=" * 70)
    print()

    # Overall counts
    print(f"Total entries: {stats['total_entries']}")
    print(f"Entries with status: {stats['entries_with_status']}")
    print(f"Entries with decision: {stats['entries_with_decision']}")
    print()

    # Status distribution
    print("-" * 70)
    print("STATUS DISTRIBUTION")
    print("-" * 70)
    status_counts = stats["status_counts"]
    for status, count in sorted(
        status_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (
            (count / stats["entries_with_status"] * 100)
            if stats["entries_with_status"] > 0
            else 0
        )
        print(f"  {status:20s}: {count:5d} ({percentage:6.2f}%)")
    print()

    # Decision distribution
    print("-" * 70)
    print("DECISION DISTRIBUTION")
    print("-" * 70)
    decision_counts = stats["decision_counts"]
    for decision, count in sorted(
        decision_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (
            (count / stats["entries_with_decision"] * 100)
            if stats["entries_with_decision"] > 0
            else 0
        )
        print(f"  {decision:20s}: {count:5d} ({percentage:6.2f}%)")
    print()


def print_high_score_no_match(high_score_cases, threshold=250):
    """Print information about no_match cases with high scores."""
    print("=" * 70)
    print(f"NO_MATCH CASES WITH MATCH SCORE > {threshold}")
    print("=" * 70)
    print()

    if not high_score_cases:
        print(f"No cases found with decision='no_match' and match_score > {threshold}")
        print()
        return

    # Calculate min and max scores
    scores = [case["match_score"] for case in high_score_cases]
    min_score = min(scores)
    max_score = max(scores)

    print(f"Found {len(high_score_cases)} cases:")
    print(f"Score range: {min_score} - {max_score}")
    print()

    # Sort by match_score descending
    for case in sorted(high_score_cases, key=lambda x: x["match_score"], reverse=True):
        print(f"Individual ID: {case['individual_id']}")
        print(f"  Match Score: {case['match_score']}")
        print(f"  Best Match Registry ID: {case['best_match_registry_id']}")
        print(f"  Status: {case['status']}")
        print(f"  Query Count: {case['query_count']}")
        if case["best_query_image"]:
            print(f"  Best Query Image: {case['best_query_image']}")
        if case["best_match_image"]:
            print(f"  Best Match Image: {case['best_match_image']}")
        print()

    print("=" * 70)
    print()


def print_proven_below_threshold(proven_cases, threshold=250):
    """Print information about proven cases with low match scores."""
    print("=" * 70)
    print(f"CONFIRMED CASES WITH MATCH SCORE < {threshold}")
    print("=" * 70)
    print()

    if not proven_cases:
        print(f"No cases found with status='confirmed' and match_score < {threshold}")
        print()
        return

    # Calculate min and max scores
    scores = [case["match_score"] for case in proven_cases]
    min_score = min(scores)
    max_score = max(scores)

    print(f"Found {len(proven_cases)} cases:")
    print(f"Score range: {min_score} - {max_score}")
    print()

    # Sort by match_score ascending (lowest first)
    for case in sorted(proven_cases, key=lambda x: x["match_score"]):
        print(f"Individual ID: {case['individual_id']}")
        print(f"  Match Score: {case['match_score']}")
        print(f"  Best Match Registry ID: {case['best_match_registry_id']}")
        print(f"  Decision: {case['decision']}")
        print(f"  Query Count: {case['query_count']}")
        if case["best_query_image"]:
            print(f"  Best Query Image: {case['best_query_image']}")
        if case["best_match_image"]:
            print(f"  Best Match Image: {case['best_match_image']}")
        print()

    print("=" * 70)
    print()


def save_statistics_json(stats, output_file="batch_predictions_stats.json"):
    """Save statistics to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {output_file}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze batch predictions statistics")
    parser.add_argument(
        "--input",
        "-i",
        default="batch_predictions.json",
        help="Input JSON file (default: batch_predictions.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file for statistics (optional)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Threshold for evaluation (optional)",
    )
    parser.add_argument(
        "--print-cases",
        action="store_true",
        default=False,
        help="Print individual cases (high score no_match, confirmed below threshold)",
    )

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.input}...")
    predictions = load_batch_predictions(args.input)

    # Calculate statistics
    print("Calculating statistics...")
    stats = get_statistics(predictions)

    # Compute score statistics for no_match and confirmed entries
    no_match_scores, confirmed_scores = compute_score_statistics(predictions)

    # Print statistics
    if not args.quiet:
        print_statistics(stats)
        print_score_statistics(no_match_scores, confirmed_scores)

        # Print prediction depth statistics
        (
            queries_by_depth,
            queries_without_rank3,
            single_pred_no_match,
            single_pred_total,
            total_queries_depth,
        ) = analyze_prediction_depth(predictions)
        print_prediction_depth(
            queries_by_depth,
            queries_without_rank3,
            single_pred_no_match,
            single_pred_total,
            total_queries_depth,
        )

        # Print top-1 statistics
        top1_confirmed, others_confirmed, total_queries = analyze_top1_statistics(
            predictions
        )
        print_top1_statistics(top1_confirmed, others_confirmed, total_queries)

        # Only print threshold-based output if threshold is specified
        if args.threshold is not None:
            # Print threshold-based statistics
            total_above_t, confirmed_above_t, rejected_above_t, confirmed_below_t = (
                analyze_threshold_statistics(predictions, args.threshold)
            )
            print_threshold_statistics(
                total_above_t,
                confirmed_above_t,
                rejected_above_t,
                confirmed_below_t,
                args.threshold,
            )

            # Print queries with multiple predictions above threshold
            queries_by_count, queries_with_multiple = (
                analyze_queries_by_predictions_above_threshold(
                    predictions, args.threshold
                )
            )
            print_queries_multiple_above_threshold(
                queries_by_count, queries_with_multiple, args.threshold
            )

            # Compare Top-1 vs Threshold methods
            (
                both_accept,
                both_reject,
                top1_accept_thresh_reject,
                top1_reject_thresh_accept,
                no_pred_above_t,
                all_rejected_above_t,
                some_confirmed_above_t,
            ) = compare_top1_vs_threshold_methods(predictions, args.threshold)
            print_method_comparison(
                both_accept,
                both_reject,
                top1_accept_thresh_reject,
                top1_reject_thresh_accept,
                no_pred_above_t,
                all_rejected_above_t,
                some_confirmed_above_t,
                args.threshold,
            )

            # Analyze and print outcome cases
            case_1, case_2, case_3, case_4, high_score_not_confirmed = (
                analyze_outcome_cases(predictions, args.threshold)
            )
            print_outcome_analysis(
                case_1, case_2, case_3, case_4, high_score_not_confirmed, args.threshold
            )

            # Compute and print precision/recall based on threshold
            tp, fp, tn, fn, precision, recall = compute_threshold_precision_recall(
                predictions, args.threshold
            )
            print_threshold_precision_recall(
                tp, fp, tn, fn, precision, recall, args.threshold
            )

            # Print FP and FN cases
            fp_cases = collect_fp_cases(predictions, args.threshold)
            print_fp_cases(fp_cases, args.threshold)
            fn_cases = collect_fn_cases(predictions, args.threshold)
            print_fn_cases(fn_cases, args.threshold)

            # Only print individual cases if --print-cases flag is set
            if args.print_cases:
                high_score_cases, proven_below_threshold = check_no_match_high_scores(
                    predictions, args.threshold
                )
                print_high_score_no_match(high_score_cases, args.threshold)
                print_proven_below_threshold(proven_below_threshold, args.threshold)

    # Save to file if requested
    if args.output:
        save_statistics_json(stats, args.output)


if __name__ == "__main__":
    main()
