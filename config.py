# config.py
from dotenv import load_dotenv
import os

load_dotenv()

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

if not ROBOFLOW_API_URL or not ROBOFLOW_API_KEY:
    raise RuntimeError("Please set ROBOFLOW_API_URL and ROBOFLOW_API_KEY in your .env file")
