# Import FastAPI and related dependencies
import base64
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

# Import dotenv for environment variables
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
# Import Gemini client and types
from google import genai
from google.genai import types
from sqlalchemy import desc
# Import SQLAlchemy components for async operations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Import routers and application-specific modules
from api.crawler import router as crawler_router
from api.logic.database import get_db, VideoAnalysis, init_db, cleanup_db
from api.logic.models import MealPlan

# Load environment variables and API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)


# Define an async lifespan context for application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    # Add any necessary cleanup here
    await cleanup_db()


# Create the FastAPI application instance
app = FastAPI(title="Vercel FastAPI", lifespan=lifespan)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the crawler router under the '/crawler' prefix
app.include_router(crawler_router, prefix="/crawler", tags=["crawler"])


# Define the root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI on Vercel!"}


# Define a simple hello endpoint
@app.get("/hello")
async def hello():
    return {"message": "Hello World"}


# Endpoint for video analysis using the Gemini API
@app.post("/analyze-video", response_model=MealPlan)
async def analyze_video(
        video: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_video:
        shutil.copyfileobj(video.file, temp_video)
        temp_video_path = temp_video.name
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(temp_video_path, "rb") as f:
        video_bytes = f.read()
        base64.b64encode(video_bytes).decode("utf-8")
    prompt = """
    Analyze this video showing food items and create a meal plan.

    Based on the food items visible in the video, generate:
    1. Recipes with detailed ingredients and instructions for the meal plans. Meal plans should be for the whole week. Monday to Sunday.
    2. A complete shopping list with all required ingredients for the meal plans if you don't have the ingredients in your kitchen. Which are not in the video.
    """
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type=video.content_type,
                        data=video_bytes
                    )
                ),
            ],
        ),
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            response_schema=MealPlan
        )
    )
    result = await db.execute(
        select(VideoAnalysis).order_by(desc(VideoAnalysis.created_at)).limit(1)
    )
    existing_analysis = result.scalars().first()
    if existing_analysis:
        await db.delete(existing_analysis)
        await db.commit()
    analysis = VideoAnalysis(
        filename=video.filename,
        content_type=video.content_type,
        prompt=prompt,
        analysis_text=response.parsed.model_dump_json(),
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    if os.path.exists(temp_video_path):
        os.unlink(temp_video_path)
    return response.parsed


# Endpoint to retrieve the latest video analysis
@app.get("/analysis", response_model=MealPlan)
async def get_latest_analysis(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(VideoAnalysis).order_by(desc(VideoAnalysis.created_at)).limit(1)
    )
    analysis = result.scalars().first()
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis found")
    return analysis.analysis_text


# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("index:app", host="0.0.0.0", port=8000)
