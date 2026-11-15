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
