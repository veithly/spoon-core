from pydantic import BaseModel


class Messages(BaseModel):
    role: str  # user, system, assistant
    content: str


class CompletionsRequest(BaseModel):
    session_id: str
    messages: list[Messages]


class SessionRequest(BaseModel):
    model_id: str
