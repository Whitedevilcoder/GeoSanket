"""
Rules vs BERT vs Hybrid, scored identically at FIELD level on the held-out test set.
    python compare.py [--data data/test.jsonl] [--model ../ai-service/models/ner]
Writes results.json and results.md (paste-ready table for the paper).
"""
import argparse, collections, json, os
from common import FIELDS, spans_to_gold, label2id
from baseline_rules import extract


def _norm(s): return " ".join(str(s).lower().replace(".", "").split())


STRICT = [False]
MONEY = ("budgetSanctioned", "budgetSpent")


def match(field, pred, gold, strict=False):
    """lenient: text fields match if one contains the other. strict: normalised exact match."""
    if field in MONEY: return abs(pred - gold) <= 0.01 * gold
    if field == "completionPercentage": return abs(pred - gold) < 0.5
    p, g = _norm(pred), _norm(gold)
    return p == g if strict else (p == g or p in g or g in p)


def entities_to_fields(text, ents):
    best = sorted(ents, key=lambda e: -e["score"])
    spans = [{"start": e["start"], "end": e["end"], "label": e["entity_group"]} for e in best]
    return spans_to_gold(text, spans)


def rules_fields(text):
    f = extract(text)["fields"]
    return {k: f.get(k) for k in FIELDS}


def hybrid(text, bert):
    """BERT first; rules fill gaps — but never re-use a money value BERT already assigned to another field
    (that was the cause of 98 spurious BUDGET fills: rules called the SPENT amount 'budget')."""
    r = rules_fields(text)
    taken = [v for k, v in bert.items() if k in MONEY and v is not None]
    out = {}
    for k in FIELDS:
        v = bert[k] if bert.get(k) is not None else r[k]
        if bert.get(k) is None and k in MONEY and v is not None and any(abs(v - t) <= 0.01 * t for t in taken): v = None
        out[k] = v
    return out


def score(preds, golds, strict=False, fields=None):
    fields = fields or FIELDS
    st = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields}
    for p, g in zip(preds, golds):
        for f in fields:
            pv, gv = p.get(f), g.get(f)
            if gv is None and pv is None: continue
            if gv is None: st[f]["fp"] += 1
            elif pv is None: st[f]["fn"] += 1
            elif match(f, pv, gv, strict): st[f]["tp"] += 1
            else: st[f]["fp"] += 1; st[f]["fn"] += 1
    return st


def by_pattern(systems, rows):
    """Micro-F1 on the two money fields, split by how BUDGET/SPENT appear in the document."""
    from audit_dataset import pattern
    pats = [pattern(r["spans"]) for r in rows]
    names = list(systems)
    print("\n-- Money-field F1 by document pattern --\n" + f"{'pattern':<16}{'n':>5}" + "".join(f"{n[:18]:>20}" for n in names))
    for p in sorted(set(pats)):
        idx = [i for i, q in enumerate(pats) if q == p]; line = f"{p:<16}{len(idx):>5}"
        for n in names:
            st = score([systems[n][i] for i in idx], [rows[i]["gold"] for i in idx], fields=list(MONEY))
            T = {k: sum(s[k] for s in st.values()) for k in ("tp", "fp", "fn")}
            line += f"{prf(**T)[2]:>20.3f}" if sum(T.values()) else f"{'-':>20}"
        print(line)


def error_report(name, preds, golds, texts, limit=40):
    """Counts missed / spurious / wrong-value per field + BUDGET<->SPENT swaps; dumps examples to errors_<name>.jsonl"""
    kinds, swaps, rows = collections.Counter(), collections.Counter(), []
    for p, g, t in zip(preds, golds, texts):
        for f in FIELDS:
            pv, gv = p.get(f), g.get(f)
            if gv is None and pv is None: continue
            if gv is None: kind = "spurious"
            elif pv is None: kind = "missed"
            elif match(f, pv, gv, STRICT[0]): continue
            else: kind = "wrong_value"
            kinds[(f, kind)] += 1
            if len(rows) < limit: rows.append({"field": f, "kind": kind, "pred": pv, "gold": gv, "text": t})
        if g.get("budgetSpent") is not None and p.get("budgetSanctioned") is not None and match("budgetSanctioned", p["budgetSanctioned"], g["budgetSpent"]):
            swaps["SPENT predicted as BUDGET"] += 1
        if g.get("budgetSanctioned") is not None and p.get("budgetSpent") is not None and match("budgetSpent", p["budgetSpent"], g["budgetSanctioned"]):
            swaps["BUDGET predicted as SPENT"] += 1
    print(f"\n-- Error analysis: {name} --")
    for (f, k), n in sorted(kinds.items(), key=lambda x: -x[1]): print(f"  {f:<22}{k:<13}{n}")
    for k, n in swaps.items(): print(f"  ** {k}: {n} docs")
    if not kinds: print("  (no errors)")
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    with open(f"errors_{safe}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarize(st):
    out = {f: prf(**s)[2] for f, s in st.items()}
    T = {k: sum(s[k] for s in st.values()) for k in ("tp", "fp", "fn")}
    P, R, F = prf(**T); out["MICRO_P"], out["MICRO_R"], out["MICRO_F1"] = P, R, F
    return out


def table(res):
    cols = FIELDS + ["MICRO_P", "MICRO_R", "MICRO_F1"]
    md = "| System | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for n, r in res.items(): md += f"| {n} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |\n"
    return md


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--model", default="../ai-service/models/ner"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    golds = [r["gold"] for r in rows]; texts = [r["text"] for r in rows]
    systems = {"Rule-based": [rules_fields(t) for t in texts]}
    if os.path.isdir(a.model):
        from transformers import pipeline
        pipe = pipeline("token-classification", model=a.model, aggregation_strategy="simple")
        bert = [entities_to_fields(t, e) for t, e in zip(texts, pipe(texts, batch_size=32))]
        systems["BERT (fine-tuned)"] = bert
        systems["Hybrid (BERT + rules)"] = [hybrid(t, b) for t, b in zip(texts, bert)]
    else:
        print(f"[!] No model at {a.model} — scoring rule-based baseline only. Train first: python train_ner.py")
    for name, p in systems.items(): error_report(name, p, golds, texts)
    if all("spans" in r for r in rows): by_pattern(systems, rows)
    lenient = {n: summarize(score(p, golds)) for n, p in systems.items()}
    strict = {n: summarize(score(p, golds, strict=True)) for n, p in systems.items()}
    md = f"### Lenient match (text fields match if one contains the other)\n{table(lenient)}\n### Strict match (exact after normalisation)\n{table(strict)}"
    print(f"\nEvaluated docs: {len(rows)}  ({a.data})\n" + md)
    json.dump({"lenient": lenient, "strict": strict}, open("results.json", "w"), indent=2); open("results.md", "w").write(md)


if __name__ == "__main__":
    main()
