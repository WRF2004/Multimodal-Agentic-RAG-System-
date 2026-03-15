"""
Async database setup using SQLAlchemy 2.0 with asyncpg.
"""

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase

logger = structlog.get_logger()


class Base(DeclarativeBase):
    pass


class DatabaseManager:
    """Manages async database connections and sessions."""

    def __init__(self, url: str, echo: bool = False, pool_size: int = 20):
        self.engine = create_async_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("database_initialized", url=url[:50] + "...")

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_created")

    async def get_session(self) -> AsyncSession:
        return self.session_factory()

    async def close(self):
        await self.engine.dispose()
        logger.info("database_closed")