from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "sqlite+aiosqlite:///./test.db"  # Fallback for development


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker as sync_sessionmaker

# Synchronous fallback for Scrapy integration
sync_engine = create_engine(DATABASE_URL.replace("sqlite+aiosqlite", "sqlite"), echo=True)
SyncSessionLocal = sync_sessionmaker(bind=sync_engine)
def get_session():
    return SyncSessionLocal()



# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


class VideoAnalysis(Base):
    __tablename__ = "video_analyses"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    prompt = Column(Text, nullable=True)
    analysis_text = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<VideoAnalysis {self.id}: {self.filename}>"

class FoodItemDB(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True, index=True)
    source_url = Column(String)
    food_name = Column(String)
    food_item_description = Column(String)
    price = Column(String)
    quantity = Column(String)

    def __repr__(self):
        return f"<FoodItem {self.id}: {self.food_name}>"

    def as_dict(self):
        return {
            "id": self.id,
            "source_url": self.source_url,
            "food_name": self.food_name,
            "food_item_description": self.food_item_description,
            "price": self.price,
            "quantity": self.quantity
        }

async def get_db():
    """Dependency for getting async DB session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize the database with tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def cleanup_db():
    """Cleanup the food_items table"""
    async with async_session() as session:
        await session.execute("DELETE FROM food_items")
        await session.commit()
