from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from db.models import Base

_DEFAULT_URL = "sqlite:///quantplat.db"


def init_db(database_url: str = _DEFAULT_URL):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(engine=None):
    if engine is None:
        engine = init_db()
    factory = sessionmaker(bind=engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
