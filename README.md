# Cytonix

Eine eigene KI-Web-App. Frontend in HTML/CSS/JavaScript, Backend in Python (FastAPI). Unterstützt zwei KI-Provider:

- **Groq** — schnelle Cloud-API mit kostenlosem Tier (empfohlen, läuft auch ohne Installation)
- **Ollama** — lokales Modell auf deinem Rechner (kein API-Key nötig)

---

## 🌐 Vom Handy aus deployen (Groq + Render)

Du brauchst **nichts zu installieren**. So geht's:

### 1. Groq-API-Key holen
- Gehe auf https://console.groq.com
- Account anlegen (kostenlos)
- Unter „API Keys" einen neuen Key erstellen → **kopieren** (nirgends speichern, nicht teilen)

### 2. Auf Render deployen
- Gehe auf https://render.com → mit GitHub einloggen
- **„New + → Blueprint"** → wähle dieses Repo (`KI-app`)
- Render erkennt `render.yaml` automatisch
- Bei der Frage nach `GROQ_API_KEY`: **dort den Key einfügen** (nur dort — niemals in den Code!)
- Deploy starten — fertig in 2–3 Minuten

Du bekommst eine URL wie `https://cytonix.onrender.com` — das ist deine live Cytonix-App.

> ⚠️ **API-Key sicher halten:** Trage den Key NUR in das Render-Dashboard ein. Schreibe ihn niemals in eine Datei, die committed wird. Falls er doch versehentlich auf GitHub landet: sofort bei Groq widerrufen und neu erstellen.

---

## 💻 Lokal starten (mit Groq)

```bash
cp .env.example .env
# Trage deinen GROQ_API_KEY in die .env-Datei ein
cd backend
pip install -r requirements.txt
./run.sh
```

Browser: http://localhost:8000

## 💻 Lokal starten (mit Ollama, ohne API-Key)

```bash
ollama pull llama3.2
ollama serve
```

In `.env`:
```
CYTONIX_PROVIDER=ollama
```

```bash
cd backend
pip install -r requirements.txt
./run.sh
```

---

## Konfiguration

| Variable | Default | Beschreibung |
|---|---|---|
| `CYTONIX_PROVIDER` | `groq` | `groq` oder `ollama` |
| `GROQ_API_KEY` | *(leer)* | Dein Groq-Key (nur bei `provider=groq`) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq-Modell |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama-Server (nur bei `provider=ollama`) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama-Modell |

## Endpoints

- `GET /api/health` — Status & aktiver Provider
- `POST /api/chat` — `{messages: [{role, content}, ...]}`

## Stack

- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Backend**: Python 3.10+, FastAPI, httpx, Pydantic, python-dotenv
- **KI**: Groq (Llama 3.3 70B) oder Ollama (lokal)
