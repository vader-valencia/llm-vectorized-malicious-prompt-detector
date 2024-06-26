import os 
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PWD = os.getenv("POSTGRES_PWD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PWD}@localhost:5432/{POSTGRES_DB}"
SYNC_DATABASE_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PWD}@localhost:5432/{POSTGRES_DB}"