import asyncio
from api.logic.database import init_db

async def create_tables():
    """Initialize the database tables."""
    print("Creating database tables...")
    await init_db()
    print("Database tables created successfully!")

if __name__ == "__main__":
    asyncio.run(create_tables()) 