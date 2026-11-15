// backend/src/routes/ai.js — tries BERT service first, falls back to the always-on rules service
const router = require("express").Router();
const { protect, adminOnly, asyncHandler } = require("../middleware/auth");

const clean = (u) => (u || "").replace(/\/$/, "");
const RULES_URL = clean(process.env.AI_SERVICE_URL); // Render: always-on, rule-based
const BERT_URL = clean(process.env.AI_BERT_URL);     // optional: laptop/Colab tunnel running the BERT model

async function call(base, engine, text, ms) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const r = await fetch(`${base}/extract?engine=${engine}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }), signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally { clearTimeout(timer); }
}

router.post("/extract", protect, adminOnly, asyncHandler(async (req, res) => {
  const text = (req.body.text || "").trim();
  if (text.length < 20) return res.status(400).json({ success: false, error: "Provide at least 20 characters of text" });

  const attempts = [];
  if (BERT_URL) attempts.push({ base: BERT_URL, engine: "hybrid", ms: 20000, label: "bert-hybrid" });
  if (RULES_URL) attempts.push({ base: RULES_URL, engine: "rules", ms: 70000, label: "rules" }); // Render cold start ≈ 50s
  if (!attempts.length) return res.status(503).json({ success: false, error: "Set AI_SERVICE_URL (and optionally AI_BERT_URL) on the backend" });

  const errors = [];
  for (const a of attempts) {
    try {
      const data = await call(a.base, a.engine, text, a.ms);
      return res.json({ success: true, engineUsed: a.label, fallback: errors.length > 0, ...data });
    } catch (e) {
      const why = e.name === "AbortError" ? "timeout" : e.message;
      console.error(`[AI] ${a.label} failed: ${why}`);
      errors.push(`${a.label}: ${why}`);
    }
  }
  res.status(502).json({ success: false, error: `All AI engines failed → ${errors.join("; ")}` });
}));

module.exports = router;
