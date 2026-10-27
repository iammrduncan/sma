import math

import pytest

from sma.compare import compare_run_dirs
from sma.errors import CompareRefused
from sma.kl import compare_dumps, kl_position, perplexity, top1_id


def _entry(token_id: int, prob: float) -> dict:
    return {"id": token_id, "logprob": math.log(prob)}


def test_page_formula_with_tail() -> None:
    # p = 0.7, 0.2, tail 0.1. q on the same ids = 0.6, 0.2, tail 0.2.
    reference = [_entry(1, 0.7), _entry(2, 0.2)]
    candidate = [_entry(1, 0.6), _entry(2, 0.2)]
    term = kl_position(reference, candidate)
    expected = 0.7 * math.log(0.7 / 0.6) + 0.2 * math.log(0.2 / 0.2) + 0.1 * math.log(0.1 / 0.2)
    assert term["method"] == "page_topk_tail"
    assert term["approximate"] is False
    assert term["k"] == 2
    assert term["kl"] == pytest.approx(expected)
    assert term["q_tail"] == pytest.approx(0.2)


def test_missing_candidate_token_is_labeled_approximate() -> None:
    reference = [_entry(1, 0.5), _entry(9, 0.4)]
    candidate = [_entry(1, 0.5), _entry(3, 0.4)]
    term = kl_position(reference, candidate)
    assert term["approximate"] is True
    assert term["method"] == "page_topk_tail_missing_folded"
    assert 9 in term["missing_ids"]
    assert term["kl"] is not None


def test_epsilon_floor_keeps_a_zero_tail_in_the_mean() -> None:
    reference = [_entry(1, 0.5)]
    candidate = [_entry(1, 1.0)]
    term = kl_position(reference, candidate)
    assert term["floored"] is True
    assert term["infinite"] is True
    assert term["kl"] == pytest.approx(0.5 * math.log(0.5 / 1.0) + 0.5 * math.log(0.5 / 1e-12))
    dumped_ref = _dump([10, 1], [reference], ppl=2.0)
    dumped_cand = _dump([10, 1], [candidate], ppl=2.0)
    compared = compare_dumps(dumped_ref, dumped_cand)
    assert compared["kl_floored_positions"] == 1
    assert compared["kl_infinite_positions"] == 1
    assert compared["kl_mean"] == pytest.approx(term["kl"])


def test_tiny_positive_mass_is_floored_but_not_infinite() -> None:
    reference = [_entry(1, 0.5)]
    candidate = [_entry(1, 1.0 - 1e-15)]
    term = kl_position(reference, candidate)
    assert term["floored"] is True
    assert term["infinite"] is False
    compared = compare_dumps(
        _dump([10, 1], [reference], ppl=2.0),
        _dump([10, 1], [candidate], ppl=2.0),
    )
    assert compared["kl_floored_positions"] == 1
    assert compared["kl_infinite_positions"] == 0


def test_perplexity_and_top1() -> None:
    assert perplexity([math.log(0.5), math.log(0.5)]) == pytest.approx(2.0)
    assert perplexity([]) is None
    assert top1_id([_entry(3, 0.2), _entry(1, 0.7)]) == 1
    # Tie goes to the smaller id.
    assert top1_id([{"id": 4, "logprob": -1.0}, {"id": 2, "logprob": -1.0}]) == 2


def _dump(token_ids: list[int], tops: list[list[dict]], ppl: float, fingerprint: str = "sha256:abc") -> dict:
    positions = []
    for index in range(1, len(token_ids)):
        positions.append(
            {
                "index": index,
                "true_token_id": token_ids[index],
                "true_logprob": tops[index - 1][0]["logprob"],
                "top": tops[index - 1],
            }
        )
    return {
        "corpus_version": "sma-corpus-v1",
        "window_tokens": len(token_ids),
        "top_k": len(tops[0]),
        "perplexity": ppl,
        "windows": [{"index": 0, "token_ids": token_ids, "positions": positions}],
        "tokenizer_fingerprint": fingerprint,
    }


def test_compare_dumps_reports_mean_kl_and_top1() -> None:
    ids = [10, 1, 2]
    reference = _dump(ids, [[_entry(1, 0.7), _entry(2, 0.2)], [_entry(2, 0.6), _entry(4, 0.3)]], ppl=2.0)
    candidate = _dump(ids, [[_entry(1, 0.6), _entry(2, 0.2)], [_entry(4, 0.6), _entry(2, 0.3)]], ppl=3.0)
    result = compare_dumps(reference, candidate)
    assert result["kl_mean"] is not None
    assert result["k"] == 2
    assert result["top1_agreement"] == pytest.approx(0.5)
    assert result["delta_ppl"] == pytest.approx(1.0)
    assert "tail" in result["approximation"]


def test_compare_refuses_a_different_tokenizer(tmp_path) -> None:
    ids = [10, 1]
    top = [[_entry(1, 0.8), _entry(2, 0.1)]]
    for name, fingerprint in (("ref", "sha256:one"), ("cand", "sha256:two")):
        run = tmp_path / name
        (run / "logprobs").mkdir(parents=True)
        (run / "manifest.json").write_text(
            '{"run_id": "%s", "tokenizer_fingerprint": "%s", "architecture": "llama", "quant_flag": "bf16", "engine": {"name": "vllm"}}'
            % (name, fingerprint),
            encoding="utf-8",
        )
        dump = _dump(ids, top, 2.0, fingerprint)
        (run / "logprobs" / "dump.json").write_text(__import__("json").dumps(dump), encoding="utf-8")
    with pytest.raises(CompareRefused, match="Tokenizers differ"):
        compare_run_dirs(tmp_path / "ref", tmp_path / "cand")


def test_clustered_mean_matches_a_hand_computed_cluster_robust_error() -> None:
    from sma.kl import clustered_mean

    # Window sums 3 and 9 over 2 + 2 values: mean 3. Residual sums: 3 - 2*3 = -3 and 9 - 2*3 = 3.
    mean, se = clustered_mean([[1.0, 2.0], [4.0, 5.0], []])
    assert mean == pytest.approx(3.0)
    assert se == pytest.approx(math.sqrt(2 / 1 * (9 + 9)) / 4)
    assert clustered_mean([[1.0, 2.0]]) == (pytest.approx(1.5), None)
    assert clustered_mean([]) == (None, None)


def test_compare_reports_llama_cpp_style_fidelity_statistics() -> None:
    from sma.kl import compare_dumps

    def window(index: int, rows: list[tuple[float, float, float]]) -> dict:
        # rows: (p of token 1 in the top-2, true-token logprob, tail mass)
        positions = []
        for pos, (p1, true_lp, _tail_mass) in enumerate(rows, start=1):
            positions.append(
                {
                    "index": pos,
                    "true_token_id": 1,
                    "true_logprob": true_lp,
                    "top": [{"id": 1, "logprob": math.log(p1)}, {"id": 2, "logprob": math.log((1 - p1) / 2)}],
                }
            )
        return {"index": index, "token_ids": [9, 1, 1], "positions": positions}

    reference = {"windows": [window(0, [(0.6, math.log(0.6), 0)] * 2), window(1, [(0.6, math.log(0.6), 0)] * 2)]}
    candidate = {"windows": [window(0, [(0.6, math.log(0.6), 0)] * 2), window(1, [(0.3, math.log(0.3), 0)] * 2)]}

    result = compare_dumps(reference, candidate)

    assert result["kl_quantiles"]["p50"] >= 0 and result["kl_quantiles"]["max"] == pytest.approx(
        max(result["kl_quantiles"].values())
    )
    assert result["kl_mean_se"] is not None and result["kl_mean_se"] > 0
    # Window 1 flips the top-1 token on both of its positions: agreement 0.5.
    assert result["top1_agreement"] == pytest.approx(0.5)
    assert result["top1_agreement_se"] == pytest.approx(0.5)
    # ln(PPL_c / PPL_r) = mean(lp_ref - lp_cand) = (0 + 0 + ln2 + ln2) / 4.
    assert result["ln_ppl_ratio"] == pytest.approx(math.log(2) / 2)
    assert result["rms_delta_p"] == pytest.approx(math.sqrt((0 + 0 + 0.09 + 0.09) / 4))
    assert "lower bound" in result["bound"]
