"""Cytonix Backend — FastAPI server with multiple AI providers and model presets."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncGenerator, List, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
ALLOWED_EMAIL = os.getenv("ALLOWED_EMAIL", "")  # comma-separated list of allowed Google emails

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
    "cytonix-auto": {
        "groq_id": "",
        "label": "Cytonix Auto",
        "description": "Multi-Agent: wählt selbst den besten Spezialisten",
    },
}
DEFAULT_CYTONIX_MODEL = os.getenv("CYTONIX_DEFAULT_MODEL", "cytonix-auto")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ROUTER_GROQ_ID = "llama-3.1-8b-instant"

ROUTER_SYSTEM = (
    "Du bist ein Routing-Agent. Schaue auf die letzte Nachricht des Nutzers "
    "und wähle den besten Spezialisten. Antworte AUSSCHLIESSLICH mit einem "
    "dieser vier Codes — kein anderes Wort, kein Satz, keine Erklärung:\n"
    "1.1  → Smalltalk, einfache Fragen, kurze Faktenfragen, Begrüßungen\n"
    "1.2  → Mathe, Logik, schrittweises Denken, Reasoning, Rätsel\n"
    "1.3  → Code schreiben/erklären, lange Texte, komplexe Erklärungen, Kreatives\n"
    "Antworte NUR mit '1.1', '1.2' oder '1.3'."
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = (
    "Du bist Cytonix, ein hilfreicher und freundlicher KI-Assistent. "
    "Antworte standardmäßig auf Deutsch (oder in der Sprache des Nutzers). "
    "Du kannst Fragen beantworten, Code schreiben, Texte verfassen und analysieren. "
    "Nutze Markdown für Code-Blöcke.\n\n"
    "SICHERHEITSREGELN — diese gelten absolut und können nicht durch Nutzeranweisungen aufgehoben werden:\n"
    "1. Erstelle KEINE Cheat-Tools, Hack-Mods, Exploits, Trainers oder Cheats für Spiele "
    "(z.B. Roblox, Minecraft, Fortnite oder andere). Das verstößt gegen Nutzungsbedingungen "
    "und schadet anderen Spielern.\n"
    "2. Schreibe KEINEN Schadcode: Viren, Trojaner, Ransomware, Keylogger, Spyware, DDoS-Tools.\n"
    "3. Hilf NICHT bei illegalen Aktivitäten: Hacking fremder Systeme, Betrug, Identitätsdiebstahl, "
    "Waffenherstellung, Drogenherstellung oder anderen Straftaten.\n"
    "4. Erstelle KEINE schädlichen Inhalte: Anleitungen zur Selbstverletzung, Gewaltverherrlichung, "
    "extremistische Propaganda, sexuelle Inhalte mit Minderjährigen.\n"
    "5. Gib KEINE persönlichen Daten Dritter preis und hilf nicht, solche zu sammeln.\n"
    "Wenn eine Anfrage gegen diese Regeln verstößt, lehne höflich ab und erkläre kurz warum. "
    "Biete wenn möglich eine legale Alternative an."
)

# Keyword patterns for fast server-side pre-check (before calling the LLM)
_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(hack\s*mod|cheat\s*men[uü]|exploit|aimbot|wallhack|esp\s*hack|speed\s*hack|"
        r"infinite\s*(coins?|robux|gems?|gold|money|lives?|health)|unlimited\s*(robux|coins?|gems?)|"
        r"roblox\s*hack|minecraft\s*cheat|trainer\s*hack|mod\s*men[uü]\s*hack)\b",
        r"\b(virus\s*schreib|trojaner|ransomware|keylogger|malware\s*code|"
        r"ddos\s*(tool|angriff|script)|botnet)\b",
        r"\b(fremde[sn]?\s*(account|passwort|konto)\s*(hack|knack|stehlen))\b",
    ]
]

def _is_blocked(text: str) -> bool:
    """Returns True if the request clearly violates safety rules."""
    for pat in _BLOCKED_PATTERNS:
        if pat.search(text):
            return True
    return False

BLOCKED_REPLY = (
    "⛔ **Das kann ich nicht machen.**\n\n"
    "Diese Anfrage verstößt gegen meine Sicherheitsregeln — ich helfe nicht bei Hacks, "
    "Cheats, Schadcode oder anderen schädlichen Inhalten.\n\n"
    "Wenn du an einem echten Spielprojekt oder Programmier-Projekt arbeitest, "
    "helfe ich dir gerne dabei auf legalem Weg."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Cytonix API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_token_cache: dict[str, float] = {}   # sha256(token) -> expiry (unix seconds)
_token_email: dict[str, str] = {}    # sha256(token) -> email


async def _verify_google_token(token: str) -> str:
    """Verify a Google ID token and return the email. Raises HTTPException on failure."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    if token_hash in _token_cache and _token_cache[token_hash] > now + 60:
        return _token_email[token_hash]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": token},
            )
    except Exception:
        raise HTTPException(status_code=401, detail="Google-Token-Prüfung fehlgeschlagen.")
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Ungültiges oder abgelaufenes Google-Token.")
    data = r.json()
    if GOOGLE_CLIENT_ID and data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Token nicht für diese App ausgestellt.")
    email = data.get("email", "").lower()
    if not email:
        raise HTTPException(status_code=401, detail="Keine E-Mail-Adresse im Token.")
    if ALLOWED_EMAIL:
        allowed = {e.strip().lower() for e in ALLOWED_EMAIL.split(",") if e.strip()}
        if email not in allowed:
            raise HTTPException(status_code=403, detail="Kein Zugriff für diese E-Mail-Adresse.")
    exp = float(data.get("exp", now + 3600))
    _token_cache[token_hash] = exp
    _token_email[token_hash] = email
    return email


async def _auth_check(authorization: str = Header(default="")) -> None:
    """FastAPI dependency — enforces Google auth if GOOGLE_CLIENT_ID is configured."""
    if not GOOGLE_CLIENT_ID:
        return  # auth disabled
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Nicht angemeldet.")
    await _verify_google_token(authorization[7:])


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Any  # plain string OR list of multimodal content parts


class ToolFlags(BaseModel):
    web: bool = False
    weather: bool = False
    calc: bool = False
    translate: bool = False
    imgedit: bool = False


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    model: Optional[str] = None
    tools: Optional[ToolFlags] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    provider: str
    routed_to: Optional[str] = None  # set when Auto mode picked a specialist


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
    if key in CYTONIX_MODELS and CYTONIX_MODELS[key]["groq_id"]:
        return key, CYTONIX_MODELS[key]["groq_id"]
    fallback = "cytonix-1.3" if DEFAULT_CYTONIX_MODEL == "cytonix-auto" else DEFAULT_CYTONIX_MODEL
    return fallback, CYTONIX_MODELS.get(fallback, {"groq_id": GROQ_MODEL_FALLBACK})["groq_id"]


def _last_user_text(messages: List["Message"]) -> str:
    """Extract the most recent user text for the router agent."""
    for m in reversed(messages):
        if m.role != "user":
            continue
        if isinstance(m.content, str):
            return m.content
        if isinstance(m.content, list):
            parts = [p.get("text", "") for p in m.content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(t for t in parts if t)
    return ""


async def _route_with_agent(user_text: str) -> str:
    """Ask a fast small model which specialist to use. Returns a cytonix model id."""
    if not user_text or not GROQ_API_KEY:
        return "cytonix-1.3"
    snippet = user_text[:600]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ROUTER_GROQ_ID,
                    "messages": [
                        {"role": "system", "content": ROUTER_SYSTEM},
                        {"role": "user", "content": snippet},
                    ],
                    "stream": False,
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return "cytonix-1.3"
    if "1.1" in choice:
        return "cytonix-1.1"
    if "1.2" in choice:
        return "cytonix-1.2"
    return "cytonix-1.3"


@app.get("/api/config")
async def api_config():
    """Public endpoint — returns frontend config (Google client ID for login)."""
    return {
        "google_client_id": GOOGLE_CLIENT_ID,
        "auth_required": bool(GOOGLE_CLIENT_ID),
    }


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


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ─────────── Tools ───────────

async def _tool_web_search(query: str) -> Optional[dict]:
    """Free web search via DuckDuckGo Instant Answer + Wikipedia fallback."""
    if not query:
        return None
    results: list[dict] = []
    summary = ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("AbstractText"):
                    summary = data["AbstractText"]
                    if data.get("AbstractURL"):
                        results.append({"title": data.get("Heading", query), "url": data["AbstractURL"], "snippet": summary[:200]})
                for topic in (data.get("RelatedTopics") or [])[:5]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({"title": topic.get("Text", "")[:80], "url": topic.get("FirstURL", ""), "snippet": topic.get("Text", "")[:240]})
    except Exception:
        pass
    # Wikipedia summary as backup if DDG empty
    if not summary:
        try:
            from urllib.parse import quote
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    f"https://de.wikipedia.org/api/rest_v1/page/summary/{quote(query)}",
                    headers={"User-Agent": "Cytonix/1.0"},
                    follow_redirects=True,
                )
                if r.status_code == 200:
                    j = r.json()
                    if j.get("extract"):
                        summary = j["extract"]
                        results.append({"title": j.get("title", query), "url": j.get("content_urls", {}).get("desktop", {}).get("page", ""), "snippet": summary[:240]})
        except Exception:
            pass
    if not summary and not results:
        return None
    return {"summary": summary, "results": results, "query": query}


async def _tool_weather(location: str) -> Optional[dict]:
    """Free weather via Open-Meteo (no API key needed)."""
    if not location:
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "de"},
            )
            if geo.status_code != 200:
                return None
            g = geo.json().get("results") or []
            if not g:
                return None
            spot = g[0]
            lat, lon, name = spot["latitude"], spot["longitude"], spot.get("name", location)
            country = spot.get("country", "")
            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                    "forecast_days": 3, "timezone": "auto",
                },
            )
            if wx.status_code != 200:
                return None
            return {"location": f"{name}, {country}", **wx.json()}
    except Exception:
        return None


_WEATHER_CODES = {
    0: "klar", 1: "fast klar", 2: "teilweise bewölkt", 3: "bewölkt",
    45: "Nebel", 48: "Reif-Nebel", 51: "leichter Sprühregen", 53: "Sprühregen",
    55: "starker Sprühregen", 61: "leichter Regen", 63: "Regen", 65: "starker Regen",
    71: "leichter Schnee", 73: "Schnee", 75: "starker Schnee", 77: "Schneegriesel",
    80: "Regenschauer", 81: "starker Regenschauer", 82: "Wolkenbruch",
    85: "Schneeschauer", 86: "starker Schneeschauer",
    95: "Gewitter", 96: "Gewitter mit Hagel", 99: "schweres Gewitter",
}


def _format_weather(w: dict) -> str:
    cur = w.get("current", {})
    daily = w.get("daily", {})
    code = cur.get("weather_code", 0)
    cond = _WEATHER_CODES.get(code, f"Code {code}")
    out = (
        f"Wetter in {w['location']} (jetzt): {cur.get('temperature_2m')}°C ({cond}), "
        f"gefühlt {cur.get('apparent_temperature')}°C, "
        f"Luftfeuchte {cur.get('relative_humidity_2m')}%, "
        f"Wind {cur.get('wind_speed_10m')} km/h."
    )
    days = daily.get("time", [])
    if days:
        out += "\nVorhersage:"
        for i, d in enumerate(days[:3]):
            tmax = daily["temperature_2m_max"][i]
            tmin = daily["temperature_2m_min"][i]
            dc = _WEATHER_CODES.get(daily["weather_code"][i], "")
            out += f"\n  {d}: {tmin}–{tmax}°C, {dc}"
    return out


# Safe calculator: only digits, spaces, basic operators, parens, decimal points
_SAFE_MATH_RE = re.compile(r"^[\d\s+\-*/().,%^]+$")

def _tool_calculator(expr: str) -> Optional[dict]:
    expr = expr.strip().replace(",", ".").replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expr or not _SAFE_MATH_RE.match(expr):
        return None
    if len(expr) > 200:
        return None
    try:
        # Only built-in ops — no names, no calls, no attributes
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 (sandboxed)
    except Exception:
        return None
    if isinstance(result, (int, float)):
        if isinstance(result, float):
            result = round(result, 10)
        return {"expression": expr, "result": result}
    return None


_LOCATION_RE = re.compile(r"(?:in|für|bei|von)\s+([A-ZÄÖÜ][\wäöüÄÖÜß\- ]{2,40})", re.IGNORECASE)
_MATH_LINE_RE = re.compile(r"[\d.,]+\s*[+\-*/×÷^][\d.,\s+\-*/×÷^()]*[\d.,)]")


def _extract_query_for_tool(text: str, tool: str) -> str:
    """Pull the most relevant snippet for a given tool from the user's prompt."""
    text = text.strip()
    if tool == "weather":
        m = _LOCATION_RE.search(text)
        if m:
            return m.group(1).strip(" ?.!,")
        # fallback: capitalised word
        cap = re.search(r"\b([A-ZÄÖÜ][\wäöüÄÖÜß\-]{2,})\b", text)
        return cap.group(1) if cap else text
    if tool == "calc":
        m = _MATH_LINE_RE.search(text)
        return m.group(0) if m else ""
    return text


async def _run_tools(user_text: str, flags: ToolFlags) -> tuple[list[dict], str]:
    """Run enabled tools, return (events_for_ui, system_context_for_llm)."""
    events: list[dict] = []
    context_blocks: list[str] = []

    if flags.calc:
        expr = _extract_query_for_tool(user_text, "calc")
        if expr:
            res = _tool_calculator(expr)
            if res:
                line = f"{res['expression']} = {res['result']}"
                events.append({"type": "tool_result", "tool": "calc", "label": "🔢 Rechner", "body": line})
                context_blocks.append(f"[Rechner]: {line}")

    if flags.weather:
        loc = _extract_query_for_tool(user_text, "weather")
        if loc:
            res = await _tool_weather(loc)
            if res:
                body = _format_weather(res)
                events.append({"type": "tool_result", "tool": "weather", "label": "🌤️ Wetter", "body": body})
                context_blocks.append(f"[Aktuelles Wetter]:\n{body}")

    if flags.web:
        res = await _tool_web_search(user_text)
        if res:
            body_lines = []
            if res.get("summary"):
                body_lines.append(res["summary"])
            for r in res.get("results", [])[:4]:
                if r.get("url"):
                    body_lines.append(f"• {r.get('title','')} — {r.get('url')}")
            body = "\n".join(body_lines)[:1500]
            events.append({"type": "tool_result", "tool": "web", "label": "🌐 Web-Suche", "body": body})
            context_blocks.append(f"[Web-Suche zu \"{res['query']}\"]:\n{body}")

    system_ctx = ""
    if context_blocks:
        system_ctx = (
            "Aktuelle Werkzeug-Ergebnisse — nutze sie als Faktengrundlage für deine Antwort, "
            "ohne sie wörtlich zu wiederholen:\n\n" + "\n\n".join(context_blocks)
        )
    return events, system_ctx


async def _stream_groq(messages: list[dict], groq_model: str) -> AsyncGenerator[str, None]:
    """Yield SSE strings from Groq streaming API, filtering <think> blocks live."""
    if not GROQ_API_KEY:
        yield _sse({"type": "error", "message": "GROQ_API_KEY fehlt."})
        return
    in_think = False
    buf = ""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": groq_model, "messages": messages, "stream": True},
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    yield _sse({"type": "error", "message": body.decode()[:300]})
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        yield _sse({"type": "done"})
                        return
                    try:
                        content = json.loads(raw)["choices"][0]["delta"].get("content") or ""
                    except Exception:
                        continue
                    if not content:
                        continue
                    buf += content
                    # Strip <think>…</think> blocks (DeepSeek-R1 reasoning traces)
                    while True:
                        if not in_think:
                            idx = buf.find("<think>")
                            if idx == -1:
                                if buf:
                                    yield _sse({"type": "chunk", "content": buf})
                                buf = ""
                                break
                            before = buf[:idx]
                            if before:
                                yield _sse({"type": "chunk", "content": before})
                            buf = buf[idx + len("<think>"):]
                            in_think = True
                            yield _sse({"type": "thinking"})
                        else:
                            idx = buf.find("</think>")
                            if idx == -1:
                                break  # still accumulating think block
                            buf = buf[idx + len("</think>"):].lstrip("\n")
                            in_think = False
                            yield _sse({"type": "thinking_done"})
    except httpx.TimeoutException:
        yield _sse({"type": "error", "message": "Groq Timeout."})
    except Exception as e:
        yield _sse({"type": "error", "message": str(e)})


@app.post("/api/chat/stream", dependencies=[Depends(_auth_check)])
async def chat_stream(req: ChatRequest):
    """Streaming endpoint — returns SSE with live chunks and agent routing events."""
    use_vision = _has_image(req.messages)
    requested = (req.model or "").lower()

    # Fast safety pre-check — block obvious harmful requests before calling the LLM
    user_text = _last_user_text(req.messages)
    if _is_blocked(user_text):
        async def _blocked() -> AsyncGenerator[str, None]:
            yield _sse({"type": "chunk", "content": BLOCKED_REPLY})
            yield _sse({"type": "done"})
            yield _sse({"type": "meta", "model": "safety-filter", "routed_to": None})
        return StreamingResponse(
            _blocked(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def generate() -> AsyncGenerator[str, None]:
        routed_to: Optional[str] = None
        effective_model = req.model

        # Run enabled tools first — emit results as SSE events and collect context
        tool_context = ""
        flags = req.tools
        if flags and any([flags.web, flags.weather, flags.calc]):
            yield _sse({"type": "tools_running", "active": [t for t in ("web","weather","calc") if getattr(flags, t, False)]})
            tool_events, tool_context = await _run_tools(user_text, flags)
            for ev in tool_events:
                yield _sse(ev)

        if requested == "cytonix-auto" and not use_vision and PROVIDER == "groq":
            yield _sse({"type": "routing"})
            routed_to = await _route_with_agent(_last_user_text(req.messages))
            effective_model = routed_to
            label = CYTONIX_MODELS.get(routed_to, {}).get("label", routed_to)
            yield _sse({"type": "routed", "routed_to": routed_to, "label": label})

        cytonix_id, groq_model = _resolve_model(effective_model, use_vision)
        sys_prompt = SYSTEM_PROMPT
        if flags and flags.translate:
            sys_prompt += "\n\nÜbersetzer-Modus aktiv: Wenn der Nutzer einen Text in eine andere Sprache übersetzen will, antworte nur mit der Übersetzung — keine Erklärungen, kein Vor- oder Nachtext, außer er fragt explizit danach."
        if tool_context:
            sys_prompt += "\n\n" + tool_context
        payload = [{"role": "system", "content": sys_prompt}]
        for m in req.messages:
            payload.append({"role": m.role, "content": m.content})

        if PROVIDER == "groq":
            async for chunk in _stream_groq(payload, groq_model):
                yield chunk
        else:
            # Ollama: blocking call, emit as single chunk
            try:
                flattened = []
                for msg in payload:
                    c = msg["content"]
                    if isinstance(c, list):
                        texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                        msg = {**msg, "content": "\n".join(texts) or "(Bild ignoriert im Ollama-Modus)"}
                    flattened.append(msg)
                reply = await _chat_ollama(flattened)
                yield _sse({"type": "chunk", "content": reply})
                yield _sse({"type": "done"})
            except HTTPException as e:
                yield _sse({"type": "error", "message": e.detail})
            except Exception as e:
                yield _sse({"type": "error", "message": str(e)})

        yield _sse({"type": "meta", "model": cytonix_id, "routed_to": routed_to})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(_auth_check)])
async def chat(req: ChatRequest):
    if _is_blocked(_last_user_text(req.messages)):
        return ChatResponse(reply=BLOCKED_REPLY, model="safety-filter", provider=PROVIDER)
    use_vision = _has_image(req.messages)

    # Auto mode: a small router agent picks the best specialist before answering.
    requested = (req.model or "").lower()
    routed_to: Optional[str] = None
    effective_model = req.model
    if requested == "cytonix-auto" and not use_vision and PROVIDER == "groq":
        routed_to = await _route_with_agent(_last_user_text(req.messages))
        effective_model = routed_to

    cytonix_id, groq_model = _resolve_model(effective_model, use_vision)

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
    return ChatResponse(reply=reply, model=active_model, provider=PROVIDER, routed_to=routed_to)


if (PROJECT_ROOT / "index.html").exists():
    @app.get("/")
    async def root():
        return FileResponse(PROJECT_ROOT / "index.html")

    app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")
