import uvicorn
from app.telephony.twilio_handler import app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,       # auto-reload on code changes
        log_level= "info",
    )
