from functools import wraps

from loguru import logger
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import  sessionmaker
from sqlalchemy.pool import QueuePool

from models.base.base import Base
from utils import DATABASE_URL


class BaseManager:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        max_overflow=10,
        pool_recycle=3600,
        pool_pre_ping=True,
    )
    metadata = MetaData()
    metadata.reflect(bind=engine)
    Session = sessionmaker(bind=engine)

    @staticmethod
    def db_session(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            session = BaseManager.Session()
            try:
                query = func(*args, **kwargs)
                if query is None:
                    return None
                if isinstance(query, list):
                    results = [session.execute(q) for q in query]
                else:
                    results = session.execute(query)
                session.commit()
                return results
            except SQLAlchemyError as e:
                logger.error(e)
                session.rollback()
                raise e
            finally:
                session.close()

        return wrapper

    def get_table(self, table_name: str) -> Table:
        return self.metadata.tables.get(table_name)

    ...
