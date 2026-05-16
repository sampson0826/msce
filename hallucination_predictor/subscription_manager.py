"""
Stripe subscription manager — checkout, webhooks, API key lifecycle.

Env vars:
  STRIPE_SECRET_KEY — Stripe secret key (sk_live_... or sk_test_...)
  STRIPE_WEBHOOK_SECRET — Stripe webhook signing secret
  STRIPE_PRO_PRICE_ID — Stripe Price ID for Pro tier ($199/mo)
  STRIPE_ENTERPRISE_PRICE_ID — Stripe Price ID for Enterprise tier ($999/mo)
  HALLU_BASE_URL — Base URL for the API (used in Stripe redirect)
"""

import os, json, time, hashlib, secrets, logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

log = logging.getLogger("hallu-stripe")

# ---- Data models ----
@dataclass
class Subscription:
    api_key: str
    tier: str  # "pro" | "enterprise"
    stripe_customer_id: str
    stripe_subscription_id: str
    status: str  # "active" | "canceled" | "past_due"
    created_at: float
    current_period_end: float

# ---- In-memory store (persisted to JSON) ----
_store: dict[str, Subscription] = {}
_store_path = Path(__file__).parent / "subscriptions.json"
_stripe = None


def _load_store():
    global _store
    if _store_path.exists():
        try:
            data = json.loads(_store_path.read_text())
            _store = {k: Subscription(**v) for k, v in data.items()}
        except Exception:
            _store = {}


def _save_store():
    _store_path.parent.mkdir(parents=True, exist_ok=True)
    _store_path.write_text(json.dumps(
        {k: {
            "api_key": v.api_key,
            "tier": v.tier,
            "stripe_customer_id": v.stripe_customer_id,
            "stripe_subscription_id": v.stripe_subscription_id,
            "status": v.status,
            "created_at": v.created_at,
            "current_period_end": v.current_period_end,
        } for k, v in _store.items()},
        indent=2,
    ))


def get_stripe():
    global _stripe
    if _stripe is None:
        import stripe
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
        _stripe = stripe
    return _stripe


def is_stripe_configured() -> bool:
    return bool(os.getenv("STRIPE_SECRET_KEY"))


# ---- Tier limits ----
TIER_LIMITS = {
    "free": {"requests_per_month": 1000, "rate_limit_per_minute": 10},
    "pro": {"requests_per_month": 50000, "rate_limit_per_minute": 60},
    "enterprise": {"requests_per_month": float("inf"), "rate_limit_per_minute": 300},
}


def get_tier_for_key(api_key: str) -> str:
    """Get subscription tier for an API key."""
    # Check Stripe subscriptions first
    for sub in _store.values():
        if sub.api_key == api_key and sub.status == "active":
            return sub.tier
    # Check env-configured keys (treated as enterprise)
    env_keys = os.getenv("HALLU_API_KEYS", "").split(",")
    if api_key in [k.strip() for k in env_keys if k.strip()]:
        return "enterprise"
    return "free"


def get_limit_for_key(api_key: str) -> dict:
    return TIER_LIMITS.get(get_tier_for_key(api_key), TIER_LIMITS["free"])


# ---- Free tier registration ----
def register_free_key(email: str = "") -> dict:
    """Generate a free-tier API key. Persisted to disk."""
    _load_store()
    api_key = "hallu_free_" + secrets.token_urlsafe(18)

    sub = Subscription(
        api_key=api_key,
        tier="free",
        stripe_customer_id="",
        stripe_subscription_id="",
        status="active",
        created_at=time.time(),
        current_period_end=time.time() + 365 * 24 * 3600,  # 1 year
    )

    _store[api_key] = sub
    _save_store()

    log.info(f"Free key registered: {api_key[:16]}... email={email}")
    return {
        "api_key": api_key,
        "tier": "free",
        "requests_per_month": TIER_LIMITS["free"]["requests_per_month"],
        "rate_limit_per_minute": TIER_LIMITS["free"]["rate_limit_per_minute"],
    }


def is_free_key(api_key: str) -> bool:
    """Check if a key is a registered free-tier key."""
    _load_store()
    return api_key in _store and _store[api_key].tier == "free"


# ---- Stripe checkout ----
def create_checkout_session(tier: str, base_url: str) -> dict:
    """Create a Stripe Checkout session for the given tier."""
    stripe = get_stripe()

    price_ids = {
        "pro": os.getenv("STRIPE_PRO_PRICE_ID", ""),
        "enterprise": os.getenv("STRIPE_ENTERPRISE_PRICE_ID", ""),
    }
    price_id = price_ids.get(tier)
    if not price_id:
        raise ValueError(f"No Stripe price ID configured for tier '{tier}'")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{base_url}/?checkout=success&tier={tier}",
        cancel_url=f"{base_url}/?checkout=canceled",
        metadata={"tier": tier},
    )
    return {"url": session.url, "session_id": session.id}


# ---- Webhook ----
def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """Handle Stripe webhook event."""
    stripe = get_stripe()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    if webhook_secret:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    else:
        event = json.loads(payload)

    event_type = event["type"]

    if event_type == "checkout.session.completed":
        return _handle_checkout_completed(event["data"]["object"])
    elif event_type == "customer.subscription.deleted":
        return _handle_subscription_deleted(event["data"]["object"])
    elif event_type == "customer.subscription.updated":
        return _handle_subscription_updated(event["data"]["object"])

    return {"status": "ignored", "event": event_type}


def _handle_checkout_completed(session: dict) -> dict:
    """Generate API key and activate subscription."""
    customer_id = session["customer"]
    subscription_id = session["subscription"]
    tier = session.get("metadata", {}).get("tier", "pro")

    # Generate API key
    api_key = "hallu_" + secrets.token_urlsafe(24)

    sub = Subscription(
        api_key=api_key,
        tier=tier,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
        status="active",
        created_at=time.time(),
        current_period_end=time.time() + 30 * 24 * 3600,
    )

    _load_store()
    _store[api_key] = sub
    _save_store()

    log.info(f"New {tier} subscription: key={api_key[:12]}... customer={customer_id}")
    return {
        "status": "activated",
        "api_key": api_key,
        "tier": tier,
    }


def _handle_subscription_deleted(subscription: dict) -> dict:
    sub_id = subscription["id"]
    _load_store()
    for key, sub in _store.items():
        if sub.stripe_subscription_id == sub_id:
            sub.status = "canceled"
            _save_store()
            log.info(f"Subscription canceled: key={key[:12]}...")
            return {"status": "canceled", "api_key": key}
    return {"status": "not_found"}


def _handle_subscription_updated(subscription: dict) -> dict:
    sub_id = subscription["id"]
    new_status = subscription.get("status", "active")
    current_period_end = subscription.get("current_period_end", time.time() + 30 * 24 * 3600)

    _load_store()
    for key, sub in _store.items():
        if sub.stripe_subscription_id == sub_id:
            sub.status = new_status
            sub.current_period_end = current_period_end
            _save_store()
            log.info(f"Subscription updated: key={key[:12]}... status={new_status}")
            return {"status": "updated", "api_key": key}
    return {"status": "not_found"}


# ---- Usage tracking ----
_usage_this_month: dict[str, int] = defaultdict(int)
_usage_minute: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(api_key: str) -> tuple[bool, str]:
    """Check if request is within rate limits. Returns (allowed, reason)."""
    tier = get_tier_for_key(api_key)
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    # Per-minute rate limit
    now = time.time()
    _usage_minute[api_key] = [t for t in _usage_minute[api_key] if now - t < 60]
    if len(_usage_minute[api_key]) >= limits["rate_limit_per_minute"]:
        return False, f"Rate limit exceeded ({limits['rate_limit_per_minute']}/min)"

    # Monthly limit
    monthly = _usage_this_month.get(api_key, 0)
    if monthly >= limits["requests_per_month"]:
        return False, f"Monthly limit exceeded ({limits['requests_per_month']})"

    _usage_minute[api_key].append(now)
    _usage_this_month[api_key] = monthly + 1
    return True, "ok"


# Initialize
_load_store()
