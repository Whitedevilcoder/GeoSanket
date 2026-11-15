# ML pipeline — NER for government project extraction

    pip install -r ../ai-service/requirements.txt -r ../ai-service/requirements-ml.txt
    python make_dataset.py --n 3000        # synthetic bootstrap data (held-out TEMPLATES for test)
    python train_ner.py                    # fine-tune bert-base-multilingual-cased  (use a GPU: Colab T4)
    python compare.py                      # Rules vs BERT vs Hybrid → results.md

Real data (what makes the evaluation meaningful):
1. Label 50+ real press releases in doccano with labels PROJECT BUDGET SPENT PROGRESS LEADER SCHEME DEPT.
2. `python convert_real.py export.jsonl data/real.jsonl`
3. Split into data/real_train.jsonl and data/test.jsonl (keep test real-only), then
   `python train_ner.py --extra_train data/real_train.jsonl`
4. `python compare.py --data data/test.jsonl`

## Hosting without paid plans
Hugging Face Docker/Gradio Spaces are paid; **model repos on the Hub are free**.
1. After training (Colab): `from huggingface_hub import notebook_login; notebook_login()` then `!python push_to_hub.py --repo YOURNAME/geosanket-ner`
2. Run the service anywhere with ≥2GB RAM (your laptop):
   `set NER_MODEL_ID=YOURNAME/geosanket-ner` → `uvicorn main:app --port 8000`
3. Expose it for the deployed site: `cloudflared tunnel --url http://localhost:8000` (copy the https URL)
4. On the Render *backend*: `AI_BERT_URL=<tunnel url>` — and keep `AI_SERVICE_URL` = the always-on rules service.
The backend tries BERT first and silently falls back to rules if the tunnel/laptop is offline.
