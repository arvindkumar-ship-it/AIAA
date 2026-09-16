"""
Main entry point for Agent API.

Run: python main.py

API will be available at http://localhost:8001
"""

import uvicorn
from api.routes import app
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Agent API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()