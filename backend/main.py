"""Cytonix Backend — FastAPI server that proxies chat requests to Ollama."""
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

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("CYTONIX_MODEL", "llama3.2")
SYSTEM_PROMPT = (
    "Du bist Cytonix, ein hilfreicher und freundlicher KI-Assistent. "
    "Antworte standardmäßig auf Deutsch (oder in der Sprache des Nutzers). "
    "Du kannst Fragen beantworten, Code schreiben, Texte verfassen und analysieren. "
    "Nutze Markdown für Code-Blöcke."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Cytonix API", version="0.1.0")

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


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = [m.get("name", "") for m in tags]
            has_model = any(n.startswith(MODEL) for n in names)
        return {"ok": True, "model": MODEL, "model_installed": has_model, "available": names}
    except Exception as e:
        return {"ok": False, "error": str(e), "model": MODEL}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        payload_messages.append({"role": m.role, "content": m.content})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": payload_messages,
                    "stream": False,
                },
            )
            if r.status_code == 404:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Ollama-Modell '{MODEL}' nicht gefunden. "
                        f"Lade es mit: ollama pull {MODEL}"
                    ),
                )
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama läuft nicht unter {OLLAMA_URL}. "
                "Starte Ollama mit `ollama serve` oder installiere es von https://ollama.com."
            ),
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama hat zu lange gebraucht.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend-Fehler: {e}")

    reply = data.get("message", {}).get("content", "").strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Leere Antwort von Ollama.")
    return ChatResponse(reply=reply, model=MODEL)


if (PROJECT_ROOT / "index.html").exists():
    @app.get("/")
    async def root():
        return FileResponse(PROJECT_ROOT / "index.html")

    app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")
