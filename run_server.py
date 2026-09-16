"""
Frontend-ready launcher.

`main.py` (as generated in the source material) starts the API with no
CORS middleware, so a browser-based frontend on a different origin/port
cannot call it. This script does not change a single line of the
original modules — it only imports the already-built `app` object and
attaches CORS middleware to it before serving, so the bundled `frontend/`
can talk to it.

Use this for the full stack demo:
    python run_server.py

Use `python main.py` if you only want the API exactly as originally written.
"""
import os

import structlog
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware

from api.routes import app

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
