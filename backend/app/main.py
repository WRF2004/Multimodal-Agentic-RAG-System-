"""
FastAPI application entry point.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import config_manager
from app.api import chat_router, documents_router, config_router, evaluation_router

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("application_starting")
    cfg = config_manager.load()
    logger.info("config_loaded", llm_model=cfg.llm.model, vectorstore=cfg.vectorstore.provider)
    yield
    logger.info("application_shutting_down")


app = FastAPI(
    title="Agentic RAG System",
    description="多模态 Agentic RAG 检索增强生成系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(config_router)
app.include_router(evaluation_router)


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/system/info")
async def system_info():
    from app.core.registry import registry
    return {
        "components": registry.list_components(),
        "config_loaded": config_manager.config is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)