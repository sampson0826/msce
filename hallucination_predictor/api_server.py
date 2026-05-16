"""
Constraint AI — Hallucination Detection API

Usage:
  python -m constraint_residual.hallucination_predictor.api_server --port 8080
  curl -X POST http://localhost:8080/detect -H "Content-Type: application/json" \
       -d '{"text": "What happens if you eat watermelon seeds?"}'
"""

import sys, os, time, json, asyncio, logging, hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from constraint_residual.hallucination_predictor.subscription_manager import (
    is_stripe_configured, create_checkout_session, handle_webhook,
    get_tier_for_key, check_rate_limit, register_free_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hallu-api")

# ---- Globals ----
wrapper = None
bank = None
request_count = 0
total_latency_ms = 0.0
_usage_by_key: dict[str, int] = defaultdict(int)

# ---- API Key Auth ----
VALID_KEYS: set[str] = set()
_api_key_enabled = False


def load_api_keys():
    global VALID_KEYS, _api_key_enabled
    key_str = os.getenv("HALLU_API_KEYS", "")
    if key_str:
        VALID_KEYS = set(k.strip() for k in key_str.split(",") if k.strip())
        _api_key_enabled = len(VALID_KEYS) > 0
        log.info(f"API key auth enabled: {len(VALID_KEYS)} key(s) configured")
    else:
        log.info("API key auth disabled (set HALLU_API_KEYS to enable)")


def _is_valid_key(api_key: str) -> bool:
    """Check if an API key is valid (env keys, Stripe-generated, or registered free)."""
    if api_key in VALID_KEYS:
        return True
    tier = get_tier_for_key(api_key)
    return tier in ("free", "pro", "enterprise")


# ---- Schema ----
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Input text to check for hallucination risk")
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Generation temperature")


class DetectResponse(BaseModel):
    text: str
    hallucination_score: float = Field(..., description="Constraint residual Δ||Π|| — positive = more likely hallucination, negative = likely factual")
    hallucination_probability: float = Field(..., description="Calibrated P(hallucination) from TruthfulQA logistic regression (AUC 0.816)")
    risk_level: str = Field(..., description="low / medium / high / critical")
    latency_ms: float
    model: str
    method: str = "constraint_residual"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    uptime_seconds: float
    requests_served: int
    avg_latency_ms: float


# ---- Detection Engine ----
# Calibrated on TruthfulQA false-premise subset (n=30, Qwen2.5-7B, AUC 0.816, Cohen's d=1.06, p=0.018)
# Raw score AUC = 0.816 (baseline_comparison_poc.json); logistic calibration is monotonic (same AUC)
# Logistic regression (no regularization, full-data fit): P(hallu|score) = 1/(1+exp(-(α + β·score)))
# 2-fold CV mean test AUC ≈ 0.76 (fold range 0.72–0.80)
_CALIB_ALPHA = -1.30
_CALIB_BETA = 16.52


def hallucination_probability(score: float) -> float:
    """Calibrated logistic probability (in-sample fit, n=30). P(hallucination | Δ||Π||)."""
    import math
    return 1.0 / (1.0 + math.exp(-(_CALIB_ALPHA + _CALIB_BETA * score)))


def score_to_risk(score: float, prob: float = None) -> str:
    """Map score + probability to risk level using calibrated thresholds."""
    if prob is None:
        prob = hallucination_probability(score)
    if prob > 0.72:
        return "critical"
    elif prob > 0.52:
        return "high"
    elif prob > 0.32:
        return "medium"
    return "low"


def detect_hallucination(text: str, temperature: float = 0.6) -> DetectResponse:
    global wrapper, bank, request_count, total_latency_ms

    t0 = time.time()

    # Generate + extract hidden states
    state = wrapper.generate_and_extract(
        prompt=text,
        max_new_tokens=64,
        temperature=temperature,
        do_sample=(temperature > 0),
    )

    # Compute constraint states and residual
    cstates = bank.compute_all(
        state.hidden_states,
        state.layer_hidden_states,
        state.attention_weights,
    )

    from constraint_residual.hallucination_predictor.constraint_functions import (
        compute_constraint_gradients, compute_residual,
    )

    gradients = compute_constraint_gradients(cstates)
    residuals, _, _ = compute_residual(gradients)

    # Filter special tokens
    special_prefixes = ('<|im_', 'system', 'user', 'assistant', '\n', '<')
    content_indices = [
        t for t in range(min(len(state.tokens), len(residuals)))
        if not any(state.tokens[t].startswith(p) for p in special_prefixes)
    ]
    input_residuals = [residuals[t] for t in content_indices if t < len(residuals)]
    input_mean = np.mean(input_residuals) if input_residuals else 0.0

    # Output residual
    response_text = state.generated_text
    output_mean = input_mean
    if response_text and len(response_text.strip()) > 5:
        try:
            out_state = wrapper.extract_output_state(response_text)
            out_cstates = bank.compute_all(
                out_state.hidden_states,
                out_state.layer_hidden_states,
                out_state.attention_weights,
            )
            out_grads = compute_constraint_gradients(out_cstates)
            out_res, _, _ = compute_residual(out_grads)
            out_filtered = [r for r in out_res if r > 1e-6]
            output_mean = np.mean(out_filtered) if out_filtered else input_mean
        except Exception:
            pass

    delta = float(output_mean - input_mean)
    prob = hallucination_probability(delta)
    latency = (time.time() - t0) * 1000

    request_count += 1
    total_latency_ms += latency

    return DetectResponse(
        text=text[:200],
        hallucination_score=round(delta, 6),
        hallucination_probability=round(prob, 4),
        risk_level=score_to_risk(delta, prob),
        latency_ms=round(latency, 1),
        model=wrapper.model_name,
        method="constraint_residual",
    )


# ---- App Lifecycle ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    global wrapper, bank

    import torch
    from constraint_residual.hallucination_predictor.model_wrapper import ModelWrapper
    from constraint_residual.hallucination_predictor.constraint_functions import ConstraintFunctionBank
    from constraint_residual.hallucination_predictor.run_poc import (
        calibrate_truth_direction, calibrate_refusal_direction,
    )

    model_name = os.getenv("HALLU_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    device = os.getenv("HALLU_DEVICE", "cuda")

    log.info(f"Loading model {model_name} on {device}...")
    wrapper = ModelWrapper(model_name=model_name, device=device)
    bank = ConstraintFunctionBank()
    calibrate_truth_direction(wrapper, bank)
    calibrate_refusal_direction(wrapper, bank)

    log.info(f"API ready — model={model_name}, device={device}, hidden_dim={wrapper.hidden_dim}")
    yield
    log.info("Shutting down API server")


# ---- App ----
app = FastAPI(
    title="Constraint AI — Hallucination Detection API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_start_time = time.time()


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # Health, landing, Stripe webhook, demo, register are public
    if request.url.path in ("/health", "/", "/detect/demo", "/register") or request.url.path.startswith("/stripe"):
        return await call_next(request)

    if _api_key_enabled:
        api_key = request.headers.get("X-API-Key", "")
        if not api_key or not _is_valid_key(api_key):
            return JSONResponse(
                content={"error": "unauthorized", "detail": "Valid X-API-Key header required. Get one at /pricing"},
                status_code=401,
                media_type="application/json",
            )

        # Rate limiting
        if request.url.path.startswith("/detect"):
            allowed, reason = check_rate_limit(api_key)
            if not allowed:
                return JSONResponse(
                    content={"error": "rate_limited", "detail": reason},
                    status_code=429,
                    media_type="application/json",
                )

    response = await call_next(request)

    # Usage tracking
    if _api_key_enabled and request.url.path.startswith("/detect"):
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            _usage_by_key[api_key] += 1

    # Rate limit headers
    if _api_key_enabled and request.url.path.startswith("/detect"):
        api_key = request.headers.get("X-API-Key", "")
        tier = get_tier_for_key(api_key) if api_key else "free"
        limits = {"free": 1000, "pro": 50000, "enterprise": 999999}
        response.headers["X-RateLimit-Limit"] = str(limits.get(tier, 1000))
        response.headers["X-RateLimit-Remaining"] = str(
            limits.get(tier, 1000) - _usage_by_key.get(api_key, 0)
        )

    return response


@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse(LANDING_HTML)


@app.get("/health", response_model=HealthResponse)
async def health():
    global wrapper, request_count, total_latency_ms
    return HealthResponse(
        status="ok",
        model_loaded=wrapper is not None and bank is not None,
        model_name=wrapper.model_name if wrapper else "",
        device=wrapper.device if wrapper else "",
        uptime_seconds=round(time.time() - _start_time, 1),
        requests_served=request_count,
        avg_latency_ms=round(total_latency_ms / max(request_count, 1), 1),
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    global wrapper
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        return detect_hallucination(req.text, req.temperature)
    except Exception as e:
        log.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(reqs: list[DetectRequest]):
    results = []
    for req in reqs:
        try:
            results.append(detect_hallucination(req.text, req.temperature))
        except Exception as e:
            results.append({"error": str(e), "text": req.text[:100]})
    return {"results": results, "n": len(results)}


# Demo rate limiter (IP-based, 5 requests per minute per IP)
_demo_ips: dict[str, list[float]] = defaultdict(list)
_DEMO_RATE_LIMIT = 5  # requests per minute
_DEMO_WINDOW = 60  # seconds


@app.post("/detect/demo", response_model=DetectResponse)
async def detect_demo(req: DetectRequest, request: Request):
    """Demo endpoint — no API key required, rate-limited by IP (5 req/min)."""
    global wrapper
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # IP rate limiting
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    _demo_ips[client_ip] = [t for t in _demo_ips[client_ip] if now - t < _DEMO_WINDOW]
    if len(_demo_ips[client_ip]) >= _DEMO_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Demo rate limit: {_DEMO_RATE_LIMIT} requests per minute. Get an API key for unlimited access.",
        )
    _demo_ips[client_ip].append(now)

    try:
        return detect_hallucination(req.text, req.temperature)
    except Exception as e:
        log.error(f"Demo detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage")
async def usage(request: Request):
    """Return usage stats for the authenticated key."""
    if not _api_key_enabled:
        return {"error": "API key auth not enabled"}
    api_key = request.headers.get("X-API-Key", "")
    tier = get_tier_for_key(api_key)
    limits = {"free": 1000, "pro": 50000, "enterprise": 999999}
    return {
        "key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:12],
        "tier": tier,
        "requests": _usage_by_key.get(api_key, 0),
        "monthly_limit": limits.get(tier, 1000),
        "total_requests_all_keys": sum(_usage_by_key.values()),
    }


# ---- Stripe ----
class CheckoutRequest(BaseModel):
    tier: str = Field("pro", description="pro | enterprise")


@app.post("/stripe/create-checkout")
async def stripe_checkout(req: CheckoutRequest, request: Request):
    """Create a Stripe Checkout session for subscription."""
    if not is_stripe_configured():
        raise HTTPException(status_code=501, detail="Stripe not configured")
    if req.tier not in ("pro", "enterprise"):
        raise HTTPException(status_code=400, detail="Invalid tier")
    try:
        base_url = os.getenv("HALLU_BASE_URL", str(request.base_url).rstrip("/"))
        result = create_checkout_session(req.tier, base_url)
        return result
    except Exception as e:
        log.error(f"Stripe checkout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    if not is_stripe_configured():
        raise HTTPException(status_code=501, detail="Stripe not configured")
    try:
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        result = handle_webhook(payload, sig_header)
        return result
    except Exception as e:
        log.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stripe/config")
async def stripe_config():
    """Return Stripe publishable key for frontend (if configured)."""
    pub_key = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    return {
        "stripe_enabled": is_stripe_configured(),
        "publishable_key": pub_key if is_stripe_configured() else None,
    }


# ---- Registration ----
class RegisterRequest(BaseModel):
    email: str = Field("", max_length=256)


@app.post("/register")
async def register(req: RegisterRequest, request: Request):
    """Register for a free API key. No auth required."""
    try:
        result = register_free_key(req.email)
        return {
            "status": "ok",
            "api_key": result["api_key"],
            "tier": result["tier"],
            "requests_per_month": result["requests_per_month"],
            "message": "Save this API key — you'll need it for all requests. Use X-API-Key header.",
        }
    except Exception as e:
        log.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Landing page (loaded from file) ----
_LANDING_PATH = Path(__file__).parent / "landing.html"
LANDING_HTML = _LANDING_PATH.read_text()

if __name__ == "__main__":
    import uvicorn

    load_api_keys()
    port = int(os.getenv("HALLU_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
