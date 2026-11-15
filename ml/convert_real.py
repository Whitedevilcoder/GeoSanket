"""Convert doccano JSONL export ({"text", "label": [[start,end,"LABEL"],...]}) into this project's format."""
import json, sys
from common import LABELS, spans_to_gold
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w", encoding="utf-8") as out:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        spans = [{"start": s, "end": e, "label": l} for s, e, l in r.get("label", []) if l in LABELS]
        out.write(json.dumps({"text": r["text"], "spans": spans, "gold": spans_to_gold(r["text"], spans)}, ensure_ascii=False) + "\n")
print("converted ->", dst)
