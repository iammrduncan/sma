"""The BFCL questions SMA keeps when its suite has a limit: an even share of every category.

Each category gets its share of the limit (largest remainder; ties by name), spread evenly through the
category in BFCL's own order (category, then index). The container's patch (sitecustomize.py here) picks
them before generating, and `sma rescore` picks the same ones from a run that answered every question.
Owner decision 2026-10-31: 2,000 of BFCL's 4,441, about 6 hours a run before. BFCL's own data gives one
id to two live_relevance questions (live_relevance_3-3-0); both are kept or both left, so 2,000 ids can be
2,001 questions.
"""


def category_and_index(item_id: str) -> tuple[str, int]:
    """BFCL's sort key: simple_20 -> (simple, 20); live_simple_19-3-15 -> (live_simple, 19)."""
    category, index = item_id.rsplit("_", 1)
    return category, int(index.split("-")[0])


def even_share(ids: list[str], limit: int) -> set[str]:
    """The ids to keep: all of them when the limit covers them."""
    by_category: dict[str, list[tuple[int, str]]] = {}
    for item in ids:
        category, index = category_and_index(item)
        by_category.setdefault(category, []).append((index, item))
    total = sum(len(entries) for entries in by_category.values())
    if limit >= total:
        return set(ids)
    quotas = {category: limit * len(entries) / total for category, entries in by_category.items()}
    counts = {category: int(quota) for category, quota in quotas.items()}
    left = limit - sum(counts.values())
    for category in sorted(by_category, key=lambda name: (counts[name] - quotas[name], name))[:left]:
        counts[category] += 1
    keep: set[str] = set()
    for category, entries in by_category.items():
        entries.sort()
        size, count = len(entries), counts[category]
        keep.update(entries[(2 * i + 1) * size // (2 * count)][1] for i in range(count))
    return keep
