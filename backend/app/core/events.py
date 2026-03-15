"""
Event system for decoupled communication between modules.
"""

import asyncio
import structlog
from typing import Any, Callable, Coroutine
from collections import defaultdict

logger = structlog.get_logger()


class EventBus:
    """Async event bus for system-wide event handling."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = defaultdict(list)
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable[..., Coroutine]):
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        self._handlers[event_type].remove(handler)

    async def publish(self, event_type: str, data: Any = None):
        handlers = self._handlers.get(event_type, [])
        tasks = [handler(data) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("event_published", event_type=event_type, handlers=len(tasks))


event_bus = EventBus()