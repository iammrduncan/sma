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
import re
from typing import Any

from sma.errors import SmaError


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


def _quantile(ordered: list[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks, as numpy's default."""
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def clustered_mean(groups: list[list[float]]) -> tuple[float | None, float | None]:
    """Mean of all values and its cluster-robust standard error, one cluster per group.

    Positions in one window share context, so they are not independent. Treating each window
    as a cluster keeps the error bar honest. The SE is None with fewer than two clusters.
    """
    filled = [group for group in groups if group]
    count = sum(len(group) for group in filled)
    if not count:
        return None, None
    mean = sum(sum(group) for group in filled) / count
    if len(filled) < 2:
        return mean, None
    spread = sum((sum(group) - len(group) * mean) ** 2 for group in filled)
    return mean, math.sqrt(len(filled) / (len(filled) - 1) * spread) / count


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
    # Per window, for clustered standard errors.
    kl_groups: list[list[float]] = []
    top1_groups: list[list[float]] = []
    log_ratio_groups: list[list[float]] = []
    delta_p_squares: list[float] = []

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
        window_top1: list[float] = []
        window_log_ratio: list[float] = []
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
            ref_true = ref_pos.get("true_logprob")
            cand_true = cand_pos.get("true_logprob")
            if isinstance(ref_true, (int, float)) and isinstance(cand_true, (int, float)):
                # ln(PPL_candidate / PPL_reference) is the mean of this. Delta p is on the true token.
                window_log_ratio.append(float(ref_true) - float(cand_true))
                delta_p_squares.append((math.exp(float(cand_true)) - math.exp(float(ref_true))) ** 2)
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
                window_top1.append(1.0 if ref_top1 == cand_top1 else 0.0)
        per_window.append(
            {
                "index": index,
                "kl_mean": (sum(window_kl) / len(window_kl)) if window_kl else None,
                "positions": len(window_kl),
            }
        )
        kl_groups.append(window_kl)
        top1_groups.append(window_top1)
        log_ratio_groups.append(window_log_ratio)

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
    kl_mean, kl_se = clustered_mean(kl_groups)
    _, top1_se = clustered_mean(top1_groups)
    log_ratio, log_ratio_se = clustered_mean(log_ratio_groups)
    ordered = sorted(finite)
    quantiles = (
        {
            "p50": _quantile(ordered, 0.50),
            "p90": _quantile(ordered, 0.90),
            "p95": _quantile(ordered, 0.95),
            "p99": _quantile(ordered, 0.99),
            "p99_9": _quantile(ordered, 0.999),
            "max": ordered[-1],
        }
        if ordered
        else None
    )
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
        "bound": (
            "Coarse-grained KL: the reference top-k ids plus one tail bucket. Merging tokens into a "
            "bucket can only lower KL (log-sum inequality), and flooring q upward only lowers a term, "
            "so kl_mean is a lower bound on full-vocabulary KL."
        ),
        "uncertainty": (
            "Standard errors (*_se) are cluster-robust with one cluster per window, because positions "
            "in a window share context. Statistics follow llama.cpp's --kl-divergence report. SMA scores "
            "every position after the first in a window; llama.cpp scores only the second half of each chunk."
        ),
        "kl_epsilon": KL_EPSILON,
        "kl_mean": kl_mean,
        "kl_mean_se": kl_se,
        "kl_quantiles": quantiles,
        "top1_agreement_se": top1_se,
        "ln_ppl_ratio": log_ratio,
        "ln_ppl_ratio_se": log_ratio_se,
        "rms_delta_p": math.sqrt(sum(delta_p_squares) / len(delta_p_squares)) if delta_p_squares else None,
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


# llama-perplexity's own summary lines (llama.cpp tools/perplexity). Δp and "Same top p" are percents.
_LLAMA_CHUNKS = re.compile(r"(?:calculating perplexity|computing) over (\d+) chunks, n_ctx=(\d+)")
_LLAMA_FINAL = re.compile(r"Final estimate: PPL = ([\d.]+)")
_LLAMA_STATS = {
    "ppl_candidate": (r"Mean PPL\(Q\)\s*:\s*([\d.]+)", 1.0),
    "ppl_base": (r"Mean PPL\(base\)\s*:\s*([\d.]+)", 1.0),
    "delta_ppl": (r"Mean PPL\(Q\)-PPL\(base\)\s*:\s*(-?[\d.]+)", 1.0),
    "kld_mean": (r"Mean\s+KLD:\s*(-?[\d.]+)", 1.0),
    "kld_median": (r"Median\s+KLD:\s*(-?[\d.]+)", 1.0),
    "kld_p99": (r"99\.0%\s+KLD:\s*(-?[\d.]+)", 1.0),
    "rms_delta_p": (r"RMS Δp\s*:\s*(-?[\d.]+)", 0.01),
    "same_top_p": (r"Same top p:\s*(-?[\d.]+)", 0.01),
}


def parse_perplexity_log(text: str) -> dict[str, Any]:
    """Numbers from a llama-perplexity run. A value the log does not print stays None.

    Each chunk of n_ctx tokens scores its second half, minus the last token: n_ctx - n_ctx // 2 - 1.
    """
    out: dict[str, Any] = {"chunks": None, "n_ctx": None, "scored_tokens": None, "ppl": None}
    found = _LLAMA_CHUNKS.findall(text)
    if found:
        chunks, n_ctx = (int(value) for value in found[-1])
        out.update(chunks=chunks, n_ctx=n_ctx, scored_tokens=chunks * (n_ctx - n_ctx // 2 - 1))
    final = _LLAMA_FINAL.findall(text)
    if final:
        out["ppl"] = float(final[-1])
    for key, (pattern, scale) in _LLAMA_STATS.items():
        matches = re.findall(pattern, text)
        out[key] = float(matches[-1]) * scale if matches else None
    if out["ppl"] is None:
        out["ppl"] = out["ppl_candidate"]
    if out["ppl"] is None:
        raise SmaError("llama-perplexity finished without a perplexity line. See logs/logprobs-llama-perplexity.log.")
    return out
