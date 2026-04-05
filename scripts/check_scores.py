#!/usr/bin/env python3
"""Sanity-check composite scoring math against live get_important_blocks results.

Replicates _compute_composite_score locally and compares against reported scores
to identify discrepancies in temporal decay, wire freshness, or other components.
"""

import math
from datetime import datetime, timezone

# ── Current weights from get_values ──────────────────────────────────────────
WEIGHTS = {
    "importance_weight": 0.3,
    "valence_weight": 0.2,
    "temporal_weight": 0.25,
    "block_wires_weight": 0.15,
    "doc_wires_weight": 0.05,
    "wire_freshness_weight": 0.05,
    "importance_ref": 10.0,
    "valence_ref": 10.0,
    "block_wires_ref": 3.0,
    "doc_wires_ref": 8.0,
    "half_life_days": 28.0,
}

DEFAULT_UNVALUATED_IMPORTANCE = 2.0

NOW = datetime(2026, 3, 31, 20, 0, tzinfo=timezone.utc)


def compute_composite(
    importance: float,
    valence: float,
    doc_age_days: float,
    block_wire_count: int,
    doc_wire_count: int,
    wire_age_days: float,
) -> dict:
    """Compute composite score and return component breakdown."""
    w = WEIGHTS
    eff_imp = importance if importance > 0 else DEFAULT_UNVALUATED_IMPORTANCE

    components = {
        "importance": w["importance_weight"] * math.tanh(eff_imp / w["importance_ref"]),
        "valence": w["valence_weight"] * math.tanh(abs(valence) / w["valence_ref"]),
        "temporal": w["temporal_weight"] * math.exp(-doc_age_days / w["half_life_days"]),
        "block_wires": w["block_wires_weight"] * math.tanh(block_wire_count / w["block_wires_ref"]),
        "doc_wires": w["doc_wires_weight"] * math.tanh(doc_wire_count / w["doc_wires_ref"]),
        "wire_freshness": w["wire_freshness_weight"] * math.exp(-wire_age_days / w["half_life_days"]),
    }
    components["total"] = round(sum(components.values()), 4)
    return components


def age_days(iso_str: str) -> float:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return max(0.0, (NOW - dt).total_seconds() / 86400.0)


# ── Blocks from get_important_blocks (post-fix) ─────────────────────────────
# Format: (label, importance, valence, block_wires, doc_wires, reported_score,
#           doc_created_iso, newest_wire_iso_or_None)

blocks = [
    ("shattering",
     3.17, -3.322, 6, 11, 0.6447,
     "2026-02-10T10:55:46Z",  # ~49d
     None),  # unknown wire age

    ("Noetic Ecosystem quote (block-c1759775)",
     0.0, 0.0, 3, 46, 0.5235,
     "2026-02-12T01:16:48Z",  # ~47d
     None),

    ("Noetic Ecosystem 'Arrogation' (block-42b97721)",
     0.0, 0.0, 3, 46, 0.5235,
     "2026-02-12T01:16:48Z",
     None),

    ("Three-Book R5 anti-singularity",
     2.585, 2.585, 1, 7, 0.5099,
     "2026-02-19T00:00:00Z",  # ~40d approx
     None),

    ("Traversing the Hyperborean",
     2.585, 2.585, 1, 4, 0.4978,
     "2026-02-10T10:55:46Z",  # ~49d approx (arkwork batch)
     None),

    ("Noetic Ecosystem Beta objection (block-d2f8e93f)",
     0.0, 0.0, 2, 46, 0.4966,
     "2026-02-12T01:16:48Z",
     None),

    ("GARDENING",
     4.088, 3.459, 12, 18, 0.4876,
     "2026-02-10T10:55:46Z",  # arkwork batch
     None),
]

print(f"{'Block':<50} {'Reported':>8} {'Computed':>8} {'Delta':>8}  Components")
print("=" * 130)

for label, imp, val, bw, dw, reported, created, wire_created in blocks:
    doc_age = age_days(created)
    # If wire age unknown, try a range
    wire_ages_to_try = [0.0, 7.0, 28.0, 47.0] if wire_created is None else [age_days(wire_created)]

    for wa in wire_ages_to_try:
        c = compute_composite(imp, val, doc_age, bw, dw, wa)
        delta = reported - c["total"]
        flag = " <<<" if abs(delta) > 0.02 else ""

        parts = " | ".join(f"{k}={v:.4f}" for k, v in c.items() if k != "total")
        wire_label = f"wire_age={wa:.0f}d"
        print(f"  {label:<48} {reported:>8.4f} {c['total']:>8.4f} {delta:>+8.4f}  {wire_label:<14} {parts}{flag}")

    print()

# ── What-if: doc_age_days=0 (the old bug) ───────────────────────────────────
print("\n--- What-if: doc_age_days=0.0 (old bug, max freshness) ---\n")
print(f"{'Block':<50} {'Reported':>8} {'Buggy':>8} {'Match?':>8}")
print("=" * 80)

for label, imp, val, bw, dw, reported, created, wire_created in blocks:
    c = compute_composite(imp, val, 0.0, bw, dw, 0.0)
    match = "YES" if abs(reported - c["total"]) < 0.005 else "no"
    print(f"  {label:<48} {reported:>8.4f} {c['total']:>8.4f} {match:>8}")
