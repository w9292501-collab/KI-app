# Cytonix

Eine eigene KI-Web-App. Frontend in HTML/CSS/JavaScript, Backend in Python (FastAPI), KI-Modell läuft lokal über [Ollama](https://ollama.com).

## Architektur

```
Browser  ──>  index.html (HTML/CSS/JS)
                  │
                  ▼
              FastAPI-Backend (Python)  ──>  Ollama  ──>  Llama 3.2
              backend/main.py               localhost:11434
```

## Setup

### 1. Ollama installieren und starten

Lade Ollama von https://ollama.com herunter, dann:

```bash
ollama pull llama3.2
ollama serve
```

### 2. Python-Backend starten

```bash
cd backend
pip install -r requirements.txt
./run.sh
```

Oder direkt:

```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend öffnen

Öffne im Browser: http://localhost:8000

(Alternativ kannst du `index.html` auch direkt öffnen — das Frontend spricht dann automatisch mit `http://localhost:8000`.)

## Konfiguration

Das Backend nutzt Umgebungsvariablen:

| Variable | Default | Beschreibung |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Adresse des Ollama-Servers |
| `CYTONIX_MODEL` | `llama3.2` | Name des Ollama-Modells |

Beispiel mit anderem Modell:

```bash
CYTONIX_MODEL=mistral ./run.sh
```

## Endpoints

- `GET /api/health` — Status & verfügbare Modelle
- `POST /api/chat` — Chat-Anfrage `{messages: [{role, content}, ...]}`

## Stack

- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Backend**: Python 3.10+ mit FastAPI, httpx, Pydantic
- **KI**: Ollama mit Llama 3.2 (austauschbar)
