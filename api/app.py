import sys
from logging import getLogger

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from api.routers import chat, models, token

logger = getLogger("api")

app = FastAPI(docs_url=None, redoc_url=None)
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

app.include_router(models.router)
app.include_router(chat.router)
app.include_router(token.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

def start(port=8000):
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
