from datetime import timedelta, datetime

import jwt
from fastapi import APIRouter
from models.base.token import TokenRequest

from utils import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY

router = APIRouter(prefix="/v1/token")

def verify_signature(address: str, signature: str) -> bool:
    return True

@router.post("/")
async def login(body: TokenRequest):
    if not verify_signature(body.address, body.signature):
        return {"error": "Invalid signature"}
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {
        "sub": body.address,
        "exp": datetime.utcnow() + access_token_expires
    }
    encoded_jwt = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": encoded_jwt, "token_type": "bearer"}