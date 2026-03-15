"""
Configuration management API.
Allows runtime configuration changes from the UI.
"""

import structlog
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.config import config_manager
from app.core.registry import registry
from app.dependencies import factory

logger = structlog.get_logger()
router = APIRouter(prefix="/api/config", tags=["Configuration"])


class ConfigUpdateRequest(BaseModel):
    session_id: str = "default"
    overrides: dict


class ConfigResponse(BaseModel):
    session_id: str
    config: dict


@router.get("/current")
async def get_current_config(session_id: str = "default") -> ConfigResponse:
    """Get the current merged configuration for a session."""
    cfg = config_manager.get_session_config(session_id)
    # Mask API keys
    masked = _mask_secrets(cfg)
    return ConfigResponse(session_id=session_id, config=masked)


@router.post("/update")
async def update_config(request: ConfigUpdateRequest) -> ConfigResponse:
    """Update session configuration (real-time, no restart needed)."""
    config_manager.set_session_override(request.session_id, request.overrides)
    # Clear component cache so new config takes effect
    factory.clear_cache()
    registry.clear_cache()

    cfg = config_manager.get_session_config(request.session_id)
    return ConfigResponse(session_id=request.session_id, config=_mask_secrets(cfg))


@router.get("/components")
async def list_available_components():
    """List all registered pluggable components."""
    return registry.list_components()


@router.post("/reset")
async def reset_config(session_id: str = "default"):
    """Reset session configuration to defaults."""
    config_manager.clear_session_override(session_id)
    factory.clear_cache()
    registry.clear_cache()
    return {"status": "ok", "message": "Configuration reset to defaults."}


def _mask_secrets(cfg: dict) -> dict:
    """Recursively mask fields containing 'key' or 'secret'."""
    import copy
    masked = copy.deepcopy(cfg)
    for key, value in masked.items():
        if isinstance(value, dict):
            masked[key] = _mask_secrets(value)
        elif isinstance(key, str) and any(s in key.lower() for s in ["key", "secret", "password", "token"]):
            if isinstance(value, str) and len(value) > 8:
                masked[key] = value[:4] + "****" + value[-4:]
    return masked