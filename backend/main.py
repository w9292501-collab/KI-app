"""Cytonix Backend — FastAPI server with multiple AI providers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal

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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = (
    "Du bist Cytonix, ein hilfreicher und freundlicher KI-Assistent. "
    "Antworte standardmäßig auf Deutsch (oder in der Sprache des Nutzers). "
    "Du kannst Fragen beantworten, Code schreiben, Texte verfassen und analysieren. "
    "Nutze Markdown für Code-Blöcke."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Cytonix API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    model: str
    provider: str


def _active_model() -> str:
    return GROQ_MODEL if PROVIDER == "groq" else OLLAMA_MODEL


@app.get("/api/health")
async def health():
    if PROVIDER == "groq":
        if not GROQ_API_KEY:
            return {
                "ok": False,
                "provider": "groq",
                "model": GROQ_MODEL,
                "error": "GROQ_API_KEY ist nicht gesetzt.",
            }
        return {"ok": True, "provider": "groq", "model": GROQ_MODEL}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        return {"ok": True, "provider": "ollama", "model": OLLAMA_MODEL}
    except Exception as e:
        return {"ok": False, "provider": "ollama", "model": OLLAMA_MODEL, "error": str(e)}


async def _chat_groq(messages: list[dict]) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY fehlt. Setze ihn als Umgebungsvariable.",
        )
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "stream": False,
                },
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
        raise HTTPException(status_code=502, detail=f"Groq-Fehler: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend-Fehler: {e}")
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unerwartete Groq-Antwort.")


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
        raise HTTPException(
            status_code=503,
            detail=f"Ollama läuft nicht unter {OLLAMA_URL}.",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama hat zu lange gebraucht.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend-Fehler: {e}")
    return data.get("message", {}).get("content", "").strip()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        payload.append({"role": m.role, "content": m.content})

    if PROVIDER == "groq":
        reply = await _chat_groq(payload)
    elif PROVIDER == "ollama":
        reply = await _chat_ollama(payload)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unbekannter CYTONIX_PROVIDER: '{PROVIDER}' (erwartet: 'groq' oder 'ollama')",
        )

    if not reply:
        raise HTTPException(status_code=502, detail="Leere Antwort vom Modell.")
    return ChatResponse(reply=reply, model=_active_model(), provider=PROVIDER)


if (PROJECT_ROOT / "index.html").exists():
    @app.get("/")
    async def root():
        return FileResponse(PROJECT_ROOT / "index.html")

    app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")
