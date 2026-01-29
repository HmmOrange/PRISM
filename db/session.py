from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.settings import db_settings

engine = create_engine(
    db_settings.sqlalchemy_url,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
