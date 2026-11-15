"""Shared label scheme + span→field conversion (used by dataset, training, comparison)."""
from baseline_rules import _money_items, PERCENT   # frozen rules-only copy: no model download on import

LABELS = ["PROJECT", "BUDGET", "SPENT", "PROGRESS", "LEADER", "SCHEME", "DEPT"]
TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
label2id = {t: i for i, t in enumerate(TAGS)}
id2label = {i: t for t, i in label2id.items()}
FIELD_OF = {"PROJECT": "name", "BUDGET": "budgetSanctioned", "SPENT": "budgetSpent",
            "PROGRESS": "completionPercentage", "LEADER": "leaderName",
            "SCHEME": "scheme", "DEPT": "department"}
FIELDS = list(FIELD_OF.values())


def _money(t):
    items = _money_items(t)
    return items[0][2] if items else None


def _pct(t):
    m = PERCENT.search(t)
    return float(m.group(1)) if m else None


def spans_to_gold(text, spans):
    """Convert char-level entity spans into the structured field dict (first span per label wins)."""
    gold = {f: None for f in FIELDS}
    for sp in spans:
        f = FIELD_OF[sp["label"]]
        if gold[f] is not None:
            continue
        t = text[sp["start"]:sp["end"]]
        if f in ("budgetSanctioned", "budgetSpent"): gold[f] = _money(t)
        elif f == "completionPercentage": gold[f] = _pct(t)
        else: gold[f] = t.strip()
    return gold


def align_labels(offsets, spans, l2i=label2id):
    """Map subword offsets → BIO label ids. Special tokens (0,0) get -100."""
    out, seen = [], set()
    for s, e in offsets:
        if s == e:
            out.append(-100); continue
        lab = "O"
        for i, sp in enumerate(spans):
            if s < sp["end"] and e > sp["start"]:
                lab = ("I-" if i in seen else "B-") + sp["label"]
                seen.add(i); break
        out.append(l2i[lab])
    return out
