from sqlalchemy import UniqueConstraint, create_engine, Column, Integer, String, MetaData, Table, DateTime, func
from databases import Database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


from env_vars_helpers import DATABASE_URL


database = Database(DATABASE_URL)
metadata = MetaData()

Base = declarative_base()

class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True)

class MaliciousPrompt(Base):
    __tablename__ = 'malicious_prompts'
    id = Column(Integer, primary_key=True)
    prompt = Column(String, unique=True)

engine = create_engine(DATABASE_URL)

async def init_db():
    Base.metadata.create_all(bind=engine)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()