"""
Fine-tune a multilingual BERT for token classification (project entity extraction).
"""
import argparse, json, os
import numpy as np, torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorForTokenClassification, set_seed)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from common import TAGS, label2id, id2label, align_labels


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            enc = tok(r["text"], truncation=True, max_length=max_len, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc["labels"] = align_labels(offs, r["spans"])
            self.items.append(dict(enc))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--data", default="data")
    ap.add_argument("--extra_train", default="")
    ap.add_argument("--out", default="../ai-service/models/ner")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model, num_labels=len(TAGS), id2label=id2label, label2id=label2id)
    
    train_rows = load(f"{a.data}/train.jsonl")
    for fp in [x for x in a.extra_train.split(",") if x]:
        train_rows += load(fp)
        
    tr = DS(train_rows, tok, a.max_len)
    va = DS(load(f"{a.data}/val.jsonl"), tok, a.max_len)
    te = DS(load(f"{a.data}/test.jsonl"), tok, a.max_len)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        yt, yp = [], []
        for pr, lb in zip(preds, p.label_ids):
            yt.append([id2label[l] for l in lb if l != -100])
            yp.append([id2label[x] for x, l in zip(pr, lb) if l != -100])
        return {"precision": precision_score(yt, yp), "recall": recall_score(yt, yp), "f1": f1_score(yt, yp)}

    args = TrainingArguments(
        output_dir="ckpt", 
        num_train_epochs=a.epochs, 
        learning_rate=a.lr, 
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs, 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True,
        metric_for_best_model="f1", 
        weight_decay=0.01, 
        warmup_steps=500, 
        report_to="none",
        fp16=torch.cuda.is_available(), 
        save_total_limit=1, 
        seed=a.seed
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=tr, 
        eval_dataset=va, 
        data_collator=DataCollatorForTokenClassification(tok), 
        compute_metrics=metrics
    )
    
    trainer.train()
    res = trainer.predict(te)
    m = metrics(res); print("TEST (held-out templates):", m)
    
    preds = np.argmax(res.predictions, axis=2)
    yt, yp = [], []
    for pr, lb in zip(preds, res.label_ids):
        yt.append([id2label[l] for l in lb if l != -100])
        yp.append([id2label[x] for x, l in zip(pr, lb) if l != -100])
    print(classification_report(yt, yp))
    
    os.makedirs(a.out, exist_ok=True)
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    json.dump({"model": a.model, "epochs": a.epochs, "test_entity_metrics": m}, open("metrics.json", "w"), indent=2)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()