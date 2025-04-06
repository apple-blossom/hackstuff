# Apply nest_asyncio to allow nested event loops
import nest_asyncio
nest_asyncio.apply()

# Import standard libraries
import logging
import os
from threading import Thread
from dotenv import load_dotenv

# Import third-party modules
import scrapy
from fastapi import APIRouter, BackgroundTasks
from google import genai
from pydantic import BaseModel
from scrapy import signals
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from sqlalchemy.orm import Session
from twisted.internet import reactor

# Import application-specific modules
from api.logic.database import FoodItemDB, get_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

# Configure Scrapy settings
settings = get_project_settings()
settings.update({
    'LOG_LEVEL': 'DEBUG',
    'ROBOTSTXT_OBEY': False,
    'COOKIES_ENABLED': True,
    'CONCURRENT_REQUESTS': 1,
    'DOWNLOAD_DELAY': 2,
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'DEFAULT_REQUEST_HEADERS': {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'de,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
})
runner = CrawlerRunner(settings)

# Global flag to indicate if a crawl is in progress
is_crawler_running = False

# Start the reactor in a background thread if not already running
if not reactor.running:
    # Define a function to run the reactor
    def run_reactor():
        reactor.run(installSignalHandlers=False)
    # Start the reactor in a daemon thread
    Thread(target=run_reactor, daemon=True).start()

# Define the schema for food items
class FoodItem(BaseModel):
    source_url: str
    food_name: str
    food_item_description: str
    price: str
    quantity: str

# List of target websites
WEBSITES = [
    "https://www.aldi-nord.de/",
    "https://www.aldi-nord.de/"
]

# Function to extract food items using Gemini API
def extract_food_items(page_text: str) -> list[FoodItem]:
    # Check that the API key is set
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return []
    # Create a Gemini API client using the environment key
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Sending request to Gemini API...")
    prompt = f"""
    Given the following TEXT, identify food items along with their details.
    For each food item, extract the following:
    - food_name: The exact name of the product as listed.
    - food_item_description: A detailed description that clearly explains what the product is. Include the product category (e.g., cheese, chocolate, bread), its form (e.g., bar, block, slice), and any distinctive characteristics.
    - price: The price of the product.
    - quantity: The quantity or size information if available.

    Return a JSON array of objects that follow this schema.

    Example:
    [
        {{
            "food_name": "Rittersport",
            "food_item_description": "A milk chocolate bar with a smooth texture and rich flavor",
            "price": "2.99 EUR",
            "quantity": "100g"
        }},
        {{
            "food_name": "Organic Bread",
            "food_item_description": "A freshly baked whole wheat bread loaf with a crunchy crust and soft interior",
            "price": "3.50 EUR",
            "quantity": "500g"
        }}
    ]

    TEXT:
    {page_text}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[FoodItem],
                "temperature": 0,
            },
        )
        logger.info(f"Gemini API response received: {response.text}")
        return response.parsed or []
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return []

# Define the Scrapy spider for food items
class FoodCrawler(scrapy.Spider):
    name = "food_crawler"

    # Generate initial requests for each website
    def start_requests(self):
        logger.info("Starting requests...")
        for url in WEBSITES:
            logger.info(f"Making request to {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse_page,
                dont_filter=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'de,en-US;q=0.9,en;q=0.8',
                }
            )

    # Parse the response from each website
    def parse_page(self, response):
        page_text = response.text
        extracted_items = extract_food_items(page_text)
        session = get_session()
        try:
            for item in extracted_items:
                db_item = FoodItemDB(
                    source_url=response.url,
                    food_name=item.food_name,
                    food_item_description=item.food_item_description,
                    price=item.price,
                    quantity=item.quantity,
                )
                session.add(db_item)
            session.commit()
        finally:
            session.close()

# Function to run the crawl without restarting the reactor
def crawl():
    global is_crawler_running
    if is_crawler_running:
        return {"message": "Crawler is already running"}
    is_crawler_running = True
    session = get_session()
    logger.info("Starting new crawl")

    # Callback for each scraped item
    def item_scraped_callback(signal, sender, item, response, spider):
        try:
            logger.info(f"Saving item to database: {item}")
            db_item = FoodItemDB(
                source_url=item['source_url'],
                food_name=item['food_name'],
                food_item_description=item['food_item_description'],
                price=item['price'],
                quantity=item['quantity'],
            )
            session.add(db_item)
            session.commit()
            logger.info(f"Item saved: {db_item}")
        except Exception as e:
            logger.error(f"Error saving item: {str(e)}")

    # Cleanup function after crawl finishes
    def cleanup(result):
        global is_crawler_running
        is_crawler_running = False
        session.close()
        logger.info("Crawler finished")
        return result

    try:
        # Clear previous food items
        session.query(FoodItemDB).delete()
        session.commit()
        logger.info("Database cleared")
        # Create and schedule the spider
        crawler_instance = runner.create_crawler(FoodCrawler)
        crawler_instance.signals.connect(item_scraped_callback, signal=signals.item_scraped)
        deferred = runner.crawl(crawler_instance)
        deferred.addBoth(cleanup)
    except Exception as e:
        logger.error(f"Crawl error: {str(e)}")
        cleanup(None)
        raise e

# Create the FastAPI router for crawler endpoints
router = APIRouter()

# Endpoint to trigger the crawl in the background
@router.get("/scrape")
async def scrape_data(background_tasks: BackgroundTasks):
    try:
        if is_crawler_running:
            return {"message": "Crawler is already running"}
        background_tasks.add_task(crawl)
        return {"message": "Scraping started in background"}
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return {"error": str(e)}

# Endpoint to check the crawler status
@router.get("/status")
def get_crawler_status():
    return {"is_running": is_crawler_running}

# Endpoint to retrieve the scraped results from the database
@router.get("/results")
def get_results():
    session = get_session()
    try:
        items = session.query(FoodItemDB).all()
        logger.info(f"Retrieved {len(items)} items")
        return {"items": [item.as_dict() for item in items]}
    finally:
        session.close()
