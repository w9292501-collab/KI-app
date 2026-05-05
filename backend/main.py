"""Cytonix Backend — FastAPI server with multiple AI providers and model presets."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PROVIDER = os.getenv("CYTONIX_PROVIDER", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

CYTONIX_MODELS = {
    "cytonix-1.1": {
        "groq_id": "llama-3.1-8b-instant",
        "label": "Cytonix 1.1",
        "description": "Schnell – ideal für kurze Antworten",
    },
    "cytonix-1.2": {
        "groq_id": "deepseek-r1-distill-llama-70b",
        "label": "Cytonix 1.2",
        "description": "Reasoning – logisches Denken",
    },
    "cytonix-1.3": {
        "groq_id": "llama-3.3-70b-versatile",
        "label": "Cytonix 1.3",
        "description": "Leistungsstark – komplexe Probleme",
    },
}
DEFAULT_CYTONIX_MODEL = os.getenv("CYTONIX_DEFAULT_MODEL", "cytonix-1.3")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = (
    "Du bist Cytonix, ein hilfreicher und freundlicher KI-Assistent. "
    "Antworte standardmäßig auf Deutsch (oder in der Sprache des Nutzers). "
    "Du kannst Fragen beantworten, Code schreiben, Texte verfassen und analysieren. "
    "Nutze Markdown für Code-Blöcke."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Cytonix API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Any  # plain string OR list of multimodal content parts


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    provider: str


def _has_image(messages: List[Message]) -> bool:
    for m in messages:
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _resolve_model(requested: Optional[str], use_vision: bool) -> tuple[str, str]:
    if use_vision:
        return "cytonix-vision", GROQ_VISION_MODEL
    key = (requested or DEFAULT_CYTONIX_MODEL).lower()
    if key in CYTONIX_MODELS:
        return key, CYTONIX_MODELS[key]["groq_id"]
    return DEFAULT_CYTONIX_MODEL, CYTONIX_MODELS.get(
        DEFAULT_CYTONIX_MODEL, {"groq_id": GROQ_MODEL_FALLBACK}
    )["groq_id"]


@app.get("/api/models")
async def models():
    return {
        "default": DEFAULT_CYTONIX_MODEL,
        "models": [
            {"id": k, **{kk: vv for kk, vv in v.items() if kk != "groq_id"}}
            for k, v in CYTONIX_MODELS.items()
        ],
    }


@app.get("/api/health")
async def health():
    if PROVIDER == "groq":
        if not GROQ_API_KEY:
            return {"ok": False, "provider": "groq", "error": "GROQ_API_KEY ist nicht gesetzt."}
        return {"ok": True, "provider": "groq", "default_model": DEFAULT_CYTONIX_MODEL}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        return {"ok": True, "provider": "ollama", "model": OLLAMA_MODEL}
    except Exception as e:
        return {"ok": False, "provider": "ollama", "model": OLLAMA_MODEL, "error": str(e)}


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


async def _chat_groq(messages: list[dict], groq_model: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY fehlt. Setze ihn als Umgebungsvariable.")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": groq_model, "messages": messages, "stream": False},
            )
            if r.status_code == 401:
                raise HTTPException(status_code=401, detail="Groq-API-Key ungültig.")
            if r.status_code == 429:
                raise HTTPException(status_code=429, detail="Groq-Limit erreicht — bitte warte kurz.")
            r.raise_for_status()
            data = r.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq hat zu lange gebraucht.")
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Groq-Fehler: {e.response.text[:300]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend-Fehler: {e}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unerwartete Groq-Antwort.")
    return _strip_thinking(content).strip()


async def _chat_ollama(messages: list[dict]) -> str:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            )
            if r.status_code == 404:
                raise HTTPException(
                    status_code=503,
                    detail=f"Ollama-Modell '{OLLAMA_MODEL}' fehlt. Lade es: ollama pull {OLLAMA_MODEL}",
                )
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Ollama läuft nicht unter {OLLAMA_URL}.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama hat zu lange gebraucht.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend-Fehler: {e}")
    return data.get("message", {}).get("content", "").strip()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    use_vision = _has_image(req.messages)
    cytonix_id, groq_model = _resolve_model(req.model, use_vision)

    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        payload.append({"role": m.role, "content": m.content})

    if PROVIDER == "groq":
        reply = await _chat_groq(payload, groq_model)
        active_model = cytonix_id
    elif PROVIDER == "ollama":
        flattened = []
        for msg in payload:
            c = msg["content"]
            if isinstance(c, list):
                texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                msg = {**msg, "content": "\n".join(texts) or "(Bild-Anhang ignoriert im Ollama-Modus)"}
            flattened.append(msg)
        reply = await _chat_ollama(flattened)
        active_model = OLLAMA_MODEL
    else:
        raise HTTPException(status_code=500, detail=f"Unbekannter CYTONIX_PROVIDER: '{PROVIDER}'")

    if not reply:
        raise HTTPException(status_code=502, detail="Leere Antwort vom Modell.")
    return ChatResponse(reply=reply, model=active_model, provider=PROVIDER)


if (PROJECT_ROOT / "index.html").exists():
    @app.get("/")
    async def root():
        return FileResponse(PROJECT_ROOT / "index.html")

    app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")
