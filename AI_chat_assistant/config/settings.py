import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

# Model Configuration
MODEL_NAME = "llama-3.3-70b-versatile"

TEMPERATURE = 0.4

MAX_TOKENS = 1500