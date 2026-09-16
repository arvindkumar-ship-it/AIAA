# Running AIAA — Full Stack

This covers the backend + the new dark-themed frontend together.
For what each module does, see `README.md` (unchanged from the source).
For exactly what was changed vs. the source material, see `CHANGES_APPLIED.md`.

## Option A — Docker (recommended, one command)

```bash
docker compose up --build
```

- Backend: http://localhost:8001
- Frontend: http://localhost:5173

## Option B — Manual

### Backend

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Full stack (CORS-enabled, for the frontend):
python run_server.py

# OR — API exactly as originally written, no CORS:
python main.py
```

Per-module tests (as in the original README):

```bash
cd memory && python test_memory.py
cd intervention && python test_intervention.py
cd style && python test_style.py      # Test 4's assertion fails by design — see CHANGES_APPLIED.md
cd autonomy && python test_autonomy.py
cd reasoning && python test_reasoning.py
cd api && python test_api.py          # requires the server already running
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173 — it talks to the backend at `VITE_API_URL`
(defaults to `http://localhost:8001`).

## API used by the frontend

- `POST /tasks` — create + execute a task
- `GET /tasks/{task_id}`
- `GET /users/{user_id}/style`
- `GET /metrics`
