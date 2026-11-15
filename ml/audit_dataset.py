"""
Shortcut audit: does the training data let a model get away with position/co-occurrence heuristics?
Reports how BUDGET and SPENT mentions are arranged in each split. A pattern that appears in test but is
rare/absent in train means the test measures generalisation the model was never taught (or a shortcut it learned).
    python audit_dataset.py
"""
import json, collections, os

def pattern(spans):
    b = [s["start"] for s in spans if s["label"] == "BUDGET"]; sp = [s["start"] for s in spans if s["label"] == "SPENT"]
    if b and sp: return "SPENT→BUDGET" if sp[0] < b[0] else "BUDGET→SPENT"
    return "BUDGET only" if b else "SPENT only" if sp else "no money"

def load(n): return [json.loads(l) for l in open(f"data/{n}.jsonl", encoding="utf-8")]

if __name__ == "__main__":
    dist = {n: collections.Counter(pattern(r["spans"]) for r in load(n)) for n in ("train", "test")}
    pats = sorted(set().union(*dist.values()))
    print(f"{'pattern':<16}{'train %':>9}{'test %':>9}")
    bad = []
    for p in pats:
        tr = 100 * dist["train"][p] / sum(dist["train"].values()); te = 100 * dist["test"][p] / sum(dist["test"].values())
        flag = "  <-- test pattern rare/absent in train" if te > 0 and tr < 3 else ""
        if flag: bad.append(p)
        print(f"{p:<16}{tr:>8.1f}%{te:>8.1f}%{flag}")
    print("\nAUDIT:", "FAIL — " + ", ".join(bad) if bad else "PASS (all test patterns are covered in training)")
