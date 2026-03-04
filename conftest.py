import os

# Set fake env vars before any app modules are imported,
# so module-level service initialization in twilio_handler doesn't fail.
os.environ.setdefault("SUPABASE_URL",        "http://localhost")
os.environ.setdefault("SUPABASE_KEY",        "test_key")
os.environ.setdefault("DEEPGRAM_API_KEY",    "test_key")
os.environ.setdefault("ANTHROPIC_API_KEY",   "test_key")
os.environ.setdefault("TWILIO_ACCOUNT_SID",  "ACtest")
os.environ.setdefault("TWILIO_AUTH_TOKEN",   "test_token")
os.environ.setdefault("TWILIO_PHONE_NUMBER", "+10000000000")
os.environ.setdefault("JINA_API_KEY",        "test_jina_key")
