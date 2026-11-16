"""Fine-tuned BERT NER inference (lazy-loaded). Needs: pip install -r requirements-ml.txt and a trained model dir."""
import os
from extractor import extract, build_response

MODEL_ID = os.environ.get("NER_MODEL_ID", "")   # e.g. "yourname/geosanket-ner" (free Hub model repo)
MODEL_DIR = os.environ.get("NER_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ner"))
FIELD_OF = {"PROJECT": "name", "BUDGET": "budgetSanctioned", "SPENT": "budgetSpent", "PROGRESS": "completionPercentage",
            "LEADER": "leaderName", "SCHEME": "scheme", "DEPT": "department"}
_pipe = None


def source() -> str:
    return MODEL_ID or MODEL_DIR


def available() -> bool:
    return bool(MODEL_ID) or os.path.isdir(MODEL_DIR)


def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("token-classification", model=source(), aggregation_strategy="simple")
    return _pipe


def _parse(field, txt):
    from extractor import _money_items, PERCENT
    if field in ("budgetSanctioned", "budgetSpent"):
        it = _money_items(txt); return it[0][2] if it else None
    if field == "completionPercentage":
        m = PERCENT.search(txt); return float(m.group(1)) if m else None
    return txt.strip() or None


def extract_ner(text: str, hybrid: bool = True) -> dict:
    """BERT entities → fields. hybrid=True keeps rule-based values (type, status, designation) and fills gaps."""
    base = extract(text)
    fields, conf = dict(base["fields"]), dict(base["confidence"])
    if not hybrid:
        fields = {k: None for k in fields}; conf = {k: 0.0 for k in conf}
    seen = set()
    for e in sorted(_load()(text), key=lambda e: -e["score"]):
        f = FIELD_OF.get(e["entity_group"])
        if not f or f in seen: continue
        v = _parse(f, text[e["start"]:e["end"]])
        if v is not None:
            fields[f] = v; conf[f] = round(float(e["score"]), 3); seen.add(f)
    # never keep a rules-derived money value that BERT assigned to a different money field
    money = ("budgetSanctioned", "budgetSpent")
    for f in money:
        other = [fields[o] for o in money if o != f and o in seen and fields.get(o) is not None]
        if f not in seen and fields.get(f) is not None and any(abs(fields[f] - v) <= 0.01 * v for v in other):
            fields[f] = None; conf[f] = 0.0
    return build_response(fields, conf)
