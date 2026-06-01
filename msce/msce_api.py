"""MSCE Product API v3.0 — Self-Doubting AI Question-Answering System.

6 heterogeneous models + 3-layer filtering + weighted integration.
Ask a question, get an answer with calibrated confidence and disagreement.
When MSCE is uncertain, it says "I don't know" instead of guessing.
"""

import time, sys, os

_MSCE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MSCE_DIR not in sys.path:
    sys.path.insert(0, _MSCE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from product_engine import run_msce, PRODUCT_CONFIG

app = FastAPI(
    title="MSCE API v3.0",
    description="Multi-model Self-doubting Cognitive Engine — answers questions with calibrated confidence",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ──

class AskRequest(BaseModel):
    question: str
    domain: str = "auto"  # math | logic | science | verbal | auto


class StrategyDetail(BaseModel):
    strategy: str
    model: str
    status: str          # selected | contributing | outlier | low_confidence | failed
    self_confidence: float = 0.0
    core_answer: str = ""
    judge_score: float = 0.0
    weight: float = 0.0
    penalties: list[str] = []


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float        # 0.0-1.0, disagreement-penalized
    disagreement: float      # 0.0-1.0, coefficient of variation of weights
    uncertain: bool          # True = MSCE says "I don't know"
    top_strategy: str
    elapsed_time: float
    strategy_details: list[StrategyDetail] = []
    judge_verdict: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, str]


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health():
    models = {
        k: v["model"] for k, v in PRODUCT_CONFIG.items()
    }
    return HealthResponse(
        status="ok",
        version="3.0.0",
        models=models,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Ask MSCE a question. Returns answer + calibrated confidence.

    When uncertain=True, MSCE is telling you it doesn't know — trust this signal.
    """
    try:
        result = run_msce(req.question, domain=req.domain)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MSCE pipeline failed: {str(exc)}")

    strategy_details = []
    for entry in result.get("reasoning_trail", []):
        strategy_details.append(StrategyDetail(
            strategy=entry["strategy"],
            model=entry["model"],
            status=entry.get("status", "unknown"),
            self_confidence=entry.get("self_confidence", 0),
            core_answer=entry.get("core_answer", "")[:200],
            judge_score=entry.get("judge_score") or 0,
            weight=entry.get("weight", 0),
            penalties=entry.get("penalties", []),
        ))

    return AskResponse(
        question=result.get("question", req.question),
        answer=result.get("top_answer", ""),
        confidence=result.get("confidence", 0.0),
        disagreement=result.get("disagreement", 0.0),
        uncertain=result.get("uncertain", False),
        top_strategy=result.get("top_strategy", ""),
        elapsed_time=result.get("elapsed_time", 0),
        strategy_details=strategy_details,
        judge_verdict=result.get("judge_verdict", ""),
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("MSCE API v3.0 — Self-Doubting AI QA System")
    print("=" * 55)
    print("Endpoint: POST /ask  — ask a question, get answer + confidence")
    print("          GET  /health — service status")
    print("Port:    8766")
    print("Docs:    http://127.0.0.1:8766/docs")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
