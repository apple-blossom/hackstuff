# Apply nest_asyncio to allow nested event loops
import nest_asyncio

nest_asyncio.apply()

# Import standard libraries
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
from twisted.internet import reactor

# Import application-specific modules
from api.logic.database import FoodItemDB, get_session

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
    'USER_AGENT': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'DEFAULT_REQUEST_HEADERS': {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'de,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
})
runner = CrawlerRunner(settings)

# Global flag indicating if a crawl is in progress
is_crawler_running = False

# Start the Twisted reactor in a background thread if not already running
if not reactor.running:
    def run_reactor() -> None:
        reactor.run(installSignalHandlers=False)


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


def extract_food_items(page_text: str) -> list[FoodItem]:
    """Use the Gemini API to extract food item details from page_text.
    Returns a list of FoodItem objects.
    """
    if not GEMINI_API_KEY:
        return []
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        Given the following TEXT, identify food items along with their details.

        For each food item, extract the following fields:
        - **food_name**: The exact name of the product as listed.
        - **food_item_description**: A detailed description that clearly explains what the product is. Translate the description into English except for brand names and product names.
        - **price**: The price of the product in EUR.
        - **quantity**: The quantity or size information if available. Convert any German measurement units or quantity terms in this field to English as specified below.

        Please apply these conversions in the quantity field (and in the description if applicable):
        - "g" → "grams"
        - "kg" → "kilograms"
        - "ml" → "milliliters"
        - "l" → "liters"
        - "Stück" → "piece" (or "pieces" when plural)
        - "Packung" → "pack"
        - "Flasche" → "bottle"
        - "Dose" → "can"

        Ensure that:
        • All units and quantity terms are fully converted to their English equivalents.
        • The food item description is translated into English without using vague or generic terms (e.g., "Various kinds" or "Verschiedene Sorten"). Instead, be specific—for example, clarify if a product like "GAZI Grill- und Pfannenkäse" is a processed cheese or a type of grillable cheese.
        • Leave brand names out of the foot item description; instead, provide a descriptive explanation of the product.
        • The output is a valid JSON array of objects following the schema below.

        Example:
        [
            {{
                "food_name": "Rittersport",
                "food_item_description": "A milk chocolate bar with a smooth texture and rich flavor",
                "price": "2.99 EUR",
                "quantity": "100 grams"
            }},
            {{
                "food_name": "Organic Bread",
                "food_item_description": "A freshly baked whole wheat bread loaf with a crunchy crust and soft interior",
                "price": "3.50 EUR",
                "quantity": "500 grams"
            }}
        ]

        TEXT:
        {page_text}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[FoodItem],
                "temperature": 0,
            },
        )
        return response.parsed or []
    except Exception as e:
        # Log error if Gemini API call fails
        print(f"Gemini API error: {str(e)}")
        return []


class FoodCrawler(scrapy.Spider):
    name = "food_crawler"

    def start_requests(self):
        """
        Yield requests for each website in the WEBSITES list.
        """
        for url in WEBSITES:
            yield scrapy.Request(
                url=url,
                callback=self.parse_page,
                dont_filter=True,
                headers={
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    ),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'de,en-US;q=0.9,en;q=0.8',
                }
            )

    def parse_page(self, response):
        """Parse the response and extract food items using Gemini API.
        Save the results in the database.
        """
        page_text = response.text
        items = extract_food_items(page_text)
        session = get_session()
        try:
            for item in items:
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


def crawl() -> dict:
    """Schedule the FoodCrawler to run.
    Returns a message if a crawl is already in progress.
    """
    global is_crawler_running
    if is_crawler_running:
        return {"message": "Crawler is already running"}
    is_crawler_running = True
    session = get_session()

    def item_scraped_callback(signal, sender, item, response, spider):
        try:
            db_item = FoodItemDB(
                source_url=item['source_url'],
                food_name=item['food_name'],
                food_item_description=item['food_item_description'],
                price=item['price'],
                quantity=item['quantity'],
            )
            session.add(db_item)
            session.commit()
        except Exception as e:
            print(f"Error saving item: {str(e)}")

    def cleanup(result):
        global is_crawler_running
        is_crawler_running = False
        session.close()
        return result

    try:
        # Clear previous items
        session.query(FoodItemDB).delete()
        session.commit()
        crawler_instance = runner.create_crawler(FoodCrawler)
        crawler_instance.signals.connect(item_scraped_callback, signal=signals.item_scraped)
        deferred = runner.crawl(crawler_instance)
        deferred.addBoth(cleanup)
        return {"message": "Scrape started"}
    except Exception as e:
        print(f"Crawl error: {str(e)}")
        cleanup(None)
        raise e


router = APIRouter()


@router.get("/scrape")
async def scrape_data(background_tasks: BackgroundTasks) -> dict:
    """Endpoint to trigger the crawl in the background."""
    if is_crawler_running:
        return {"message": "Crawler is already running"}
    background_tasks.add_task(crawl)
    return {"message": "Scrape started in background"}


@router.get("/status")
def get_crawler_status() -> dict:
    """Endpoint to check if a crawl is currently running."""
    return {"is_running": is_crawler_running}


@router.get("/results")
def get_results() -> dict:
    """Endpoint to retrieve scraped results from the database."""
    session = get_session()
    try:
        items = session.query(FoodItemDB).all()
        return {"items": [item.as_dict() for item in items]}
    finally:
        session.close()
