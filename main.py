import asyncio
from sqlalchemy import text
from app.routes import app
from database.database import init_db, AsyncSessionLocal, get_async_db
from database.malicious_embedding_manager import MaliciousEmbeddingManager

async def initialize_embedding_manager():
    try:
        embedding_manager = MaliciousEmbeddingManager()
        await embedding_manager.initialize()
        await embedding_manager.load_initial_prompts() 
        return embedding_manager
    except Exception as e:
        print(f"Error initializing embedding manager: {e}")
        raise

async def execute_sql_script(session, filename: str):
    with open(filename, 'r') as file:
        sql_commands = file.read()
    await session.execute(text(sql_commands))
    await session.commit()

async def initialize_database():
    try:
        embedding_manager = None
        async with AsyncSessionLocal() as session:
            await init_db()
            print("created db ok")
            await execute_sql_script(session, 'enable_pgvector.sql')
            print("enabled pgvector")
            embedding_manager = await initialize_embedding_manager()
            print("initialized embedding manager")
        app.config['embedding_manager'] = embedding_manager
    except Exception as e:
        print(f"Failed to initialize the database due to: {str(e)}")

def setup_app():
    asyncio.run(initialize_database())

if __name__ == '__main__':
    setup_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
