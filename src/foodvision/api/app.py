"""
FoodVision 1.0 FastAPI Application
Production ASGI web service mounting static assets and REST endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from foodvision.config import config, STATIC_DIR
from foodvision.logger import logger
from foodvision.api.routes import router as api_router

app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="Production-grade AI Food Classification, Nutrition Profiling & Local LLM Diagnosis"
)

# Enable CORS for external / mobile integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router)

# Mount static web UI assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serves the primary camera-enabled web and mobile UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "FoodVision 1.0 Backend Running. Static UI initializing."}


@app.on_event("startup")
async def startup_event():
    logger.info("FoodVision 1.0 Server starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FoodVision 1.0 Server shutting down.")
