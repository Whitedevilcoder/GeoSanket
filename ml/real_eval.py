"""
Build a REAL-DATA test set with minimal labelling effort (field values only — no span annotation needed).

  1) put each press release in its own .txt file in raw/            (copy-paste from PIB / Delhi govt sites; note the URL)
  2) python real_eval.py prefill raw real_labels.csv                  → CSV pre-filled by the extractor (BERT if model exists, else rules)
  3) open real_labels.csv in Excel/Sheets, FIX every wrong cell, fill 'source'  (blank = not stated in the text)
  4) python real_eval.py build real_labels.csv data/real_test.jsonl
  5) python compare.py --data data/real_test.jsonl

Units in the CSV: budgets in CRORE rupees, completion as a number (e.g. 75).
"""
import csv, glob, json, os, sys

COLS = ["id", "source", "name", "budgetSanctioned_cr", "budgetSpent_cr", "completionPercentage", "leaderName", "scheme", "department", "text"]


def prefill(raw_dir, out_csv, model_dir="../ai-service/models/ner"):
    from compare import rules_fields, entities_to_fields, hybrid
    files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not files: sys.exit(f"No .txt files in {raw_dir}/")
    texts = [" ".join(open(f, encoding="utf-8", errors="ignore").read().split()) for f in files]
    if os.path.isdir(model_dir):
        from transformers import pipeline
        pipe = pipeline("token-classification", model=model_dir, aggregation_strategy="simple")
        preds = [hybrid(t, entities_to_fields(t, e)) for t, e in zip(texts, pipe(texts, batch_size=16))]
        how = "BERT+rules"
    else:
        preds = [rules_fields(t) for t in texts]; how = "rules"
    cr = lambda v: "" if v is None else f"{v / 1e7:g}"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for f, t, p in zip(files, texts, preds):
            w.writerow({"id": os.path.basename(f), "source": "", "name": p["name"] or "", "budgetSanctioned_cr": cr(p["budgetSanctioned"]),
                        "budgetSpent_cr": cr(p["budgetSpent"]), "completionPercentage": "" if p["completionPercentage"] is None else f"{p['completionPercentage']:g}",
                        "leaderName": p["leaderName"] or "", "scheme": p["scheme"] or "", "department": p["department"] or "", "text": t})
    print(f"Wrote {len(files)} rows to {out_csv} (pre-filled by {how}). Now CORRECT every cell by reading the text, then run: build")


def build(in_csv, out_jsonl):
    num = lambda s: float(s.strip()) if s and s.strip() else None
    txt = lambda s: s.strip() if s and s.strip() else None
    n = 0; missing_src = 0
    with open(in_csv, encoding="utf-8-sig", newline="") as fh, open(out_jsonl, "w", encoding="utf-8") as out:
        for r in csv.DictReader(fh):
            if not (r.get("text") or "").strip(): continue
            b, s = num(r["budgetSanctioned_cr"]), num(r["budgetSpent_cr"])
            gold = {"name": txt(r["name"]), "budgetSanctioned": b * 1e7 if b is not None else None, "budgetSpent": s * 1e7 if s is not None else None,
                    "completionPercentage": num(r["completionPercentage"]), "leaderName": txt(r["leaderName"]),
                    "scheme": txt(r["scheme"]), "department": txt(r["department"])}
            out.write(json.dumps({"text": r["text"], "gold": gold, "source": r.get("source", "")}, ensure_ascii=False) + "\n"); n += 1
            missing_src += not (r.get("source") or "").strip()
    print(f"Built {n} labelled documents -> {out_jsonl}")
    if n < 30: print(f"[!] Only {n} documents — aim for 40-50 for a defensible result.")
    if missing_src: print(f"[!] {missing_src} rows have no 'source' — record where each text came from (needed for the paper).")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "prefill": prefill(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "build": build(sys.argv[2], sys.argv[3])
    else: print(__doc__)
