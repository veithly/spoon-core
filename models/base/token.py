from pydantic import BaseModel

class TokenRequest(BaseModel):
    address: str
    signature: str