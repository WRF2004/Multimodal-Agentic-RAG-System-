"""
Message broker implementation using RabbitMQ (aio-pika).
Handles async document processing queue.
"""

import json
import structlog
from typing import Callable, Coroutine
from app.core.interfaces import MessageBrokerInterface

logger = structlog.get_logger()


class RabbitMQBroker(MessageBrokerInterface):
    """RabbitMQ message broker using aio-pika."""

    def __init__(self, url: str = "amqp://guest:guest@localhost:5672/"):
        self._url = url
        self._connection = None
        self._channel = None

    async def connect(self):
        import aio_pika
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=10)
        logger.info("rabbitmq_connected", url=self._url[:30] + "...")

    async def close(self):
        if self._connection:
            await self._connection.close()
            logger.info("rabbitmq_disconnected")

    async def publish(self, queue: str, message: dict) -> None:
        import aio_pika
        if not self._channel:
            await self.connect()
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message, ensure_ascii=False).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue,
        )
        logger.debug("message_published", queue=queue)

    async def consume(self, queue: str, callback: Callable[..., Coroutine]) -> None:
        if not self._channel:
            await self.connect()
        q = await self._channel.declare_queue(queue, durable=True)

        async def on_message(message):
            async with message.process():
                try:
                    data = json.loads(message.body.decode())
                    await callback(data)
                except Exception as e:
                    logger.error("message_processing_error", queue=queue, error=str(e))

        await q.consume(on_message)
        logger.info("consumer_started", queue=queue)


class InMemoryBroker(MessageBrokerInterface):
    """In-memory broker for development/testing (no RabbitMQ needed)."""

    def __init__(self):
        self._queues: dict[str, list] = {}
        self._callbacks: dict[str, list] = {}

    async def publish(self, queue: str, message: dict) -> None:
        if queue not in self._queues:
            self._queues[queue] = []
        self._queues[queue].append(message)

        # Immediately process if consumers exist
        for callback in self._callbacks.get(queue, []):
            try:
                await callback(message)
            except Exception as e:
                logger.error("inmemory_broker_error", error=str(e))

    async def consume(self, queue: str, callback: Callable[..., Coroutine]) -> None:
        if queue not in self._callbacks:
            self._callbacks[queue] = []
        self._callbacks[queue].append(callback)