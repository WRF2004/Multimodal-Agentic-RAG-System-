"""
Chat API endpoints - handles conversation with streaming support.
"""

import json
import uuid
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.config import config_manager
from app.dependencies import factory
from app.core.interfaces import Message

logger = structlog.get_logger()
router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    config_overrides: Optional[dict] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict] = []
    steps: list[dict] = []
    metadata: dict = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Synchronous chat endpoint."""
    # Get merged config
    if request.config_overrides:
        config_manager.set_session_override(request.session_id, request.config_overrides)
    cfg = config_manager.get_session_config(request.session_id)

    # Build components
    components = factory.build_all(cfg)
    agent = components["agent"]
    conv_manager = components["conversation_manager"]

    # Get history and resolve coreference
    history = await conv_manager.get_history(request.session_id)
    resolved_query = await conv_manager.resolve_coreference(request.message, history)

    # Save user message
    await conv_manager.add_message(
        request.session_id,
        Message(role="user", content=request.message)
    )

    # Run agent
    response = await agent.run(resolved_query, history, session_config=cfg)

    # Save assistant message
    await conv_manager.add_message(
        request.session_id,
        Message(role="assistant", content=response.answer)
    )

    # Compress history if needed
    await conv_manager.compress_history(request.session_id)

    return ChatResponse(
        session_id=request.session_id,
        answer=response.answer,
        sources=[
            {"content": s.content[:200], "metadata": s.metadata}
            for s in response.sources
        ],
        steps=[
            {
                "tool": step.action.tool,
                "input": step.action.tool_input,
                "reasoning": step.action.reasoning,
                "observation": step.observation[:300],
            }
            for step in response.steps
        ],
        metadata=response.metadata,
    )


@router.websocket("/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()
    logger.info("ws_connected", session_id=session_id)

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            config_overrides = data.get("config_overrides", None)

            if config_overrides:
                config_manager.set_session_override(session_id, config_overrides)
            cfg = config_manager.get_session_config(session_id)

            components = factory.build_all(cfg)
            agent = components["agent"]
            conv_manager = components["conversation_manager"]

            history = await conv_manager.get_history(session_id)
            resolved_query = await conv_manager.resolve_coreference(message, history)

            await conv_manager.add_message(session_id, Message(role="user", content=message))

            # Stream agent response
            full_answer = ""
            async for event in agent.run_stream(resolved_query, history, session_config=cfg):
                await websocket.send_json(event)
                if event.get("type") == "answer":
                    full_answer = event.get("content", "")
                elif event.get("type") == "token":
                    full_answer += event.get("content", "")

            await conv_manager.add_message(
                session_id,
                Message(role="assistant", content=full_answer)
            )
            await conv_manager.compress_history(session_id)

    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("ws_error", session_id=session_id, error=str(e))
        await websocket.close(code=1011)