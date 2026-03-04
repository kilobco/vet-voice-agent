import os
from dotenv import load_dotenv

load_dotenv()

# Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")

# Jina AI (used for query embedding via API — no local model needed)
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Twilio
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
