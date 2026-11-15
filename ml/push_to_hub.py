"""Upload the trained model to a FREE Hugging Face model repo (model repos stay free; only Docker/Gradio Spaces are paid).
Colab:  from huggingface_hub import notebook_login; notebook_login()
        !python push_to_hub.py --repo YOUR_USERNAME/geosanket-ner
"""
import argparse, json, os
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--dir", default="../ai-service/models/ner")
ap.add_argument("--private", action="store_true"); a = ap.parse_args()
if not os.path.isdir(a.dir): raise SystemExit(f"No trained model at {a.dir} — run train_ner.py first")
m = json.load(open("metrics.json")) if os.path.exists("metrics.json") else {}
card = f"""---
language: [en, hi]
base_model: {m.get('model', 'bert-base-multilingual-cased')}
pipeline_tag: token-classification
---
# GeoSanket NER
Extracts PROJECT, BUDGET, SPENT, PROGRESS, LEADER, SCHEME, DEPT from government project announcements.

Held-out-template test (entity-level): {json.dumps(m.get('test_entity_metrics', {}))}
"""
open(os.path.join(a.dir, "README.md"), "w").write(card)
api = HfApi(); api.create_repo(a.repo, exist_ok=True, private=a.private)
api.upload_folder(folder_path=a.dir, repo_id=a.repo)
print(f"Done -> https://huggingface.co/{a.repo}   (use NER_MODEL_ID={a.repo})")
