from sqlalchemy import UniqueConstraint, create_engine, Column, Integer, String, MetaData, Table, DateTime, func
from databases import Database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session


from env_vars_helpers import DATABASE_URL, SYNC_DATABASE_URL


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

# Change to asynchronous engine
engine = create_async_engine(DATABASE_URL)
sync_engine = create_engine(SYNC_DATABASE_URL)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Change to asynchronous session maker
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
SessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)

async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

