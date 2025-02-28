from typing import AsyncGenerator

from api.core.config import settings
from loguru import logger
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    poolclass=QueuePool,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    logger.info(f"connect: {dbapi_connection}")
    
pool_stats = {
    "checked_in": lambda: engine.pool.checkedin(),
    "checked_out": lambda: engine.pool.checkedout(),
    "size": lambda: engine.pool.size(),
    "overflow": lambda: engine.pool.overflow(),
}

