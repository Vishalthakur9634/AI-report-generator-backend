import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "ai-report-generation"

# We will initialize client and db lazily inside the proxies
client = None
db = None

def get_client_and_db():
    global client, db
    if client is None:
        if "mongodb+srv" in MONGODB_URI:
            client = AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())
        else:
            client = AsyncIOMotorClient(MONGODB_URI)
        db = client[DB_NAME]
    return client, db

class MotorCollectionProxy:
    def __init__(self, name):
        self._name = name
        
    @property
    def _collection(self):
        _, db_instance = get_client_and_db()
        return db_instance[self._name]
        
    def __getattr__(self, name):
        return getattr(self._collection, name)

class GridFSProxy:
    @property
    def _fs(self):
        _, db_instance = get_client_and_db()
        return AsyncIOMotorGridFSBucket(db_instance)
        
    def __getattr__(self, name):
        return getattr(self._fs, name)

# Collections
users_collection = MotorCollectionProxy("users")
centers_collection = MotorCollectionProxy("centers")
templates_collection = MotorCollectionProxy("templates")
reports_collection = MotorCollectionProxy("reports")

# GridFS for storing files
fs = GridFSProxy()

async def init_db():
    # Force initialization in active loop
    get_client_and_db()
    # Create indexes for performance and uniqueness
    try:
        await users_collection.create_index("email", unique=True)
        await centers_collection.create_index("name", unique=True)
    except Exception as e:
        print(f"Warning: Could not create unique indexes (likely duplicate data exists): {e}")
    
def get_db():
    _, db_instance = get_client_and_db()
    return db_instance
