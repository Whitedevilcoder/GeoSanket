import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from extractor import extract
import ner_model

app = FastAPI(title="GeoSanket AI Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "GeoSanket AI service running", "endpoints": ["POST /extract?engine=rules|bert|hybrid", "POST /extract-pdf"], "bert_model_configured": ner_model.available(), "bert_source": ner_model.source() if ner_model.available() else None}


def _run(text: str, engine: str):
    if engine == "rules":
        return extract(text)
    if engine in ("bert", "hybrid"):
        if not ner_model.available():
            raise HTTPException(503, f"No NER model configured ({ner_model.source()}). Set NER_MODEL_ID or train one (ml/train_ner.py), or use engine=rules")
        return ner_model.extract_ner(text, hybrid=(engine == "hybrid"))
    raise HTTPException(400, "engine must be rules | bert | hybrid")


@app.post("/extract")
def extract_text(req: ExtractRequest, engine: str = "rules"):
    text = req.text.strip()
    if len(text) < 20:
        raise HTTPException(400, "text must be at least 20 characters")
    return _run(text[:20000], engine)


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...), engine: str = "rules"):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a .pdf file")
    data = await file.read()
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:15])
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")
    if len(text.strip()) < 20:
        raise HTTPException(422, "No extractable text in PDF (scanned image PDFs need OCR)")
    return _run(text[:20000], engine)
