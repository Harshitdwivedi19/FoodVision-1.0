"""
FoodVision 1.0 - Main Application Launcher
Runs the production Uvicorn server hosting the FastAPI backend and Camera Web UI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn
from foodvision.config import config
from foodvision.logger import logger


def main():
    logger.info("Starting FoodVision 1.0 on %s:%d", config.host, config.port)
    uvicorn.run(
        "foodvision.api.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
