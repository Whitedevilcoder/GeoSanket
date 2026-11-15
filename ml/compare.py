"""
Rules vs BERT vs Hybrid, scored identically at FIELD level on the held-out test set.
    python compare.py [--data data/test.jsonl] [--model ../ai-service/models/ner]
Writes results.json and results.md (paste-ready table for the paper).
"""
import argparse, json, os
from common import FIELDS, spans_to_gold, label2id
from extractor import extract
from evaluate import match


def entities_to_fields(text, ents):
    best = sorted(ents, key=lambda e: -e["score"])
    spans = [{"start": e["start"], "end": e["end"], "label": e["entity_group"]} for e in best]
    return spans_to_gold(text, spans)


def rules_fields(text):
    f = extract(text)["fields"]
    return {k: f.get(k) for k in FIELDS}


def hybrid(text, bert):
    r = rules_fields(text)
    return {k: (bert[k] if bert.get(k) is not None else r[k]) for k in FIELDS}


def score(preds, golds):
    st = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    for p, g in zip(preds, golds):
        for f in FIELDS:
            pv, gv = p.get(f), g.get(f)
            if gv is None and pv is None: continue
            if gv is None: st[f]["fp"] += 1
            elif pv is None: st[f]["fn"] += 1
            elif match(f, pv, gv): st[f]["tp"] += 1
            else: st[f]["fp"] += 1; st[f]["fn"] += 1
    return st


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarize(st):
    out = {f: prf(**s)[2] for f, s in st.items()}
    T = {k: sum(s[k] for s in st.values()) for k in ("tp", "fp", "fn")}
    P, R, F = prf(**T); out["MICRO_P"], out["MICRO_R"], out["MICRO_F1"] = P, R, F
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--model", default="../ai-service/models/ner"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    golds = [r["gold"] for r in rows]; texts = [r["text"] for r in rows]
    systems = {"Rule-based": [rules_fields(t) for t in texts]}
    if os.path.isdir(a.model):
        from transformers import pipeline
        pipe = pipeline("token-classification", model=a.model, aggregation_strategy="simple")
        bert = [entities_to_fields(t, pipe(t)) for t in texts]
        systems["BERT (fine-tuned)"] = bert
        systems["Hybrid (BERT + rules)"] = [hybrid(t, b) for t, b in zip(texts, bert)]
    else:
        print(f"[!] No model at {a.model} — scoring rule-based baseline only. Train first: python train_ner.py")
    res = {name: summarize(score(p, golds)) for name, p in systems.items()}
    cols = FIELDS + ["MICRO_P", "MICRO_R", "MICRO_F1"]
    md = "| System | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for n, r in res.items(): md += f"| {n} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |\n"
    print(f"\nHeld-out docs: {len(rows)}\n" + md)
    json.dump(res, open("results.json", "w"), indent=2); open("results.md", "w").write(md)


if __name__ == "__main__":
    main()
