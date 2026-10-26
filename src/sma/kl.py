"""Top-k KL, perplexity, and top-1 agreement.

The page formula, with natural logs (vLLM returns natural-log probabilities):

    KL ≈ Σ_(i in top-k) p_i · log(p_i / q_i) + p_tail · log(p_tail / q_tail)
    PPL = exp( − mean log q(next token) )

p is the reference distribution. q_tail is 1 minus q's probability mass on the
same k token ids. When a reference token is missing from q's stored top-k, the
position is labeled approximate: that token's p mass is folded into p_tail and
q_i is not invented.
"""

from __future__ import annotations

import math
from typing import Any


def probability(logprob: float) -> float:
    if logprob != logprob:  # NaN
        return 0.0
    prob = math.exp(logprob)
    if prob < 0 or prob != prob or prob == math.inf:
        return 0.0
    return prob


def perplexity(true_logprobs: list[float]) -> float | None:
    if not true_logprobs:
        return None
    mean = sum(true_logprobs) / len(true_logprobs)
    if mean < -80:
        return None
    return math.exp(-mean)


def top1_id(entries: list[dict[str, Any]]) -> int | None:
    if not entries:
        return None
    best = max(entries, key=lambda item: (float(item["logprob"]), -int(item["id"])))
    return int(best["id"])


# Positions that would have been infinite KL stay in the mean, floored here.
KL_EPSILON = 1e-12


def _tail(mass: float) -> float:
    if mass < 0 and mass > -1e-9:
        return 0.0
    if mass < 0:
        return 0.0
    if mass > 1:
        return 1.0
    return mass


def kl_position(
    reference_top: list[dict[str, Any]],
    candidate_top: list[dict[str, Any]],
) -> dict[str, Any]:
    """KL of reference || candidate on the reference top-k, plus a tail bucket."""
    ref = [(int(item["id"]), probability(float(item["logprob"]))) for item in reference_top]
    cand = {int(item["id"]): probability(float(item["logprob"])) for item in candidate_top}
    k = len(ref)
    p_sum = sum(prob for _, prob in ref)
    p_tail = _tail(1.0 - p_sum)
    missing = [token_id for token_id, _ in ref if token_id not in cand]
    approximate = bool(missing)
    if missing:
        folded = sum(prob for token_id, prob in ref if token_id not in cand)
        used = [(token_id, prob) for token_id, prob in ref if token_id in cand]
        p_tail = _tail(p_tail + folded)
        method = "page_topk_tail_missing_folded"
    else:
        used = ref
        method = "page_topk_tail"
    q_sum = sum(cand[token_id] for token_id, _ in used)
    q_tail = _tail(1.0 - q_sum)
    floored = False
    infinite = False
    kl = 0.0
    for token_id, p_mass in used:
        if p_mass == 0:
            continue
        q_mass = cand[token_id]
        if q_mass <= 0:
            infinite = True
        if q_mass < KL_EPSILON:
            q_mass = KL_EPSILON
            floored = True
        kl += p_mass * math.log(p_mass / q_mass)
    if p_tail > 0:
        q_tail_used = q_tail
        if q_tail_used <= 0:
            infinite = True
        if q_tail_used < KL_EPSILON:
            q_tail_used = KL_EPSILON
            floored = True
        kl += p_tail * math.log(p_tail / q_tail_used)
    return {
        "kl": kl,
        "infinite": infinite,
        "floored": floored,
        "approximate": approximate,
        "method": method,
        "k": k,
        "missing_ids": missing,
        "p_tail": p_tail,
        "q_tail": q_tail,
        "epsilon": KL_EPSILON,
    }


def _windows(dump: dict[str, Any]) -> list[dict[str, Any]]:
    windows = dump.get("windows")
    if not isinstance(windows, list):
        raise ValueError("logprob dump has no windows")
    return windows


def compare_dumps(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare two stored logprob dumps. Caller checks tokenizer identity first."""
    ref_windows = _windows(reference)
    cand_windows = {int(item["index"]): item for item in _windows(candidate)}
    k_values: set[int] = set()
    methods: set[str] = set()
    finite: list[float] = []
    floored_positions = 0
    infinite_positions = 0
    approximate_positions = 0
    top1_hit = 0
    top1_total = 0
    positions = 0
    per_window: list[dict[str, Any]] = []
    token_mismatches = 0

    for ref_window in ref_windows:
        index = int(ref_window["index"])
        other = cand_windows.get(index)
        if other is None:
            continue
        ref_ids = list(ref_window.get("token_ids") or [])
        other_ids = list(other.get("token_ids") or [])
        if ref_ids != other_ids:
            token_mismatches += 1
            continue
        ref_positions = list(ref_window.get("positions") or [])
        other_positions = {int(item["index"]): item for item in other.get("positions") or []}
        window_kl: list[float] = []
        for ref_pos in ref_positions:
            pos_index = int(ref_pos["index"])
            cand_pos = other_positions.get(pos_index)
            if cand_pos is None:
                continue
            if int(ref_pos["true_token_id"]) != int(cand_pos["true_token_id"]):
                token_mismatches += 1
                continue
            term = kl_position(list(ref_pos.get("top") or []), list(cand_pos.get("top") or []))
            positions += 1
            k_values.add(int(term["k"]))
            methods.add(str(term["method"]))
            if term["approximate"]:
                approximate_positions += 1
            if term.get("floored"):
                floored_positions += 1
            if term.get("infinite"):
                infinite_positions += 1
            if term["kl"] is None:
                continue
            finite.append(float(term["kl"]))
            window_kl.append(float(term["kl"]))
            ref_top1 = top1_id(list(ref_pos.get("top") or []))
            cand_top1 = top1_id(list(cand_pos.get("top") or []))
            if ref_top1 is not None and cand_top1 is not None:
                top1_total += 1
                if ref_top1 == cand_top1:
                    top1_hit += 1
        per_window.append(
            {
                "index": index,
                "kl_mean": (sum(window_kl) / len(window_kl)) if window_kl else None,
                "positions": len(window_kl),
            }
        )

    ref_ppl = reference.get("perplexity")
    cand_ppl = candidate.get("perplexity")
    delta = None
    if isinstance(ref_ppl, (int, float)) and isinstance(cand_ppl, (int, float)):
        delta = float(cand_ppl) - float(ref_ppl)
    if len(methods) == 1:
        method = next(iter(methods))
    elif not methods:
        method = "page_topk_tail"
    else:
        method = "page_topk_tail_mixed"
    k_recorded = next(iter(k_values)) if len(k_values) == 1 else None
    return {
        "schema_version": 1,
        "method": method,
        "log_base": "e",
        "k": k_recorded,
        "k_values": sorted(k_values),
        "approximation": (
            "Top-k KL plus one tail bucket. q_tail = 1 - q mass on the reference "
            "top-k ids when those ids are present. Missing ids are folded into the "
            "tail and counted in approximate_positions. Probabilities below "
            f"{KL_EPSILON} are floored so those positions stay in the mean. "
            "This is not full-vocabulary KL."
        ),
        "kl_epsilon": KL_EPSILON,
        "kl_mean": (sum(finite) / len(finite)) if finite else None,
        "kl_positions": positions,
        "kl_finite_positions": len(finite),
        "kl_floored_positions": floored_positions,
        "kl_infinite_positions": infinite_positions,
        "kl_approximate_positions": approximate_positions,
        "top1_agreement": (top1_hit / top1_total) if top1_total else None,
        "top1_positions": top1_total,
        "perplexity_reference": ref_ppl if isinstance(ref_ppl, (int, float)) else None,
        "perplexity_candidate": cand_ppl if isinstance(cand_ppl, (int, float)) else None,
        "delta_ppl": delta,
        "windows_compared": len(per_window),
        "token_sequence_mismatches": token_mismatches,
        "per_window": per_window,
    }
