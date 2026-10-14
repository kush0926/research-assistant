"""Auth (JWT), password hashing, per-user rate limiting, and prompt-injection defences (Part 23)."""
import hashlib
import hmac
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import User, get_db

_bearer = HTTPBearer(auto_error=False)


# ---------- passwords (PBKDF2-HMAC-SHA256) ----------
def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 200_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ---------- JWT ----------
def create_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "exp": datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer), db: Session = Depends(get_db)) -> User:
    if not creds:
        raise HTTPException(401, "Not authenticated")
    try:
        data = jwt.decode(creds.credentials, settings.JWT_SECRET, algorithms=["HS256"])
        user = db.get(User, int(data["sub"]))
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    if not user:
        raise HTTPException(401, "User not found")
    return user


# ---------- rate limiting (per user, sliding 60s window, for expensive endpoints) ----------
_hits: dict[int, deque] = defaultdict(deque)


def rate_limit(user: User = Depends(current_user)) -> User:
    now = time.time()
    q = _hits[user.id]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Rate limit exceeded — try again in a minute")
    q.append(now)
    return user


# ---------- prompt-injection defences ----------
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|messages)",
    r"disregard (all |any )?(the )?(previous|prior|above|system) (instructions|prompts?)",
    r"(reveal|print|show|output|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in) ",
    r"\bsystem prompt\b",
    r"(do not|don't) (tell|inform) the user",
    r"<\s*/?\s*(system|assistant|instructions?)\s*>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.I)


def sanitize_document_text(text: str) -> tuple[str, bool]:
    """Neutralise instruction-like text embedded in ingested documents. Returns (clean_text, was_flagged)."""
    flagged = bool(_INJECTION_RE.search(text))
    if flagged:
        text = _INJECTION_RE.sub("[removed: instruction-like text]", text)
    return text, flagged


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "paper.pdf")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]
    return name or "paper.pdf"
