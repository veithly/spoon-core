import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from spoon_ai.agents.spoon_react import SpoonReactAI

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = None

@router.post("/chat")
async def chat(request: ChatRequest):
    react_agent = SpoonReactAI()
    for message in request.messages:
        react_agent.add_message(message.role, message.content)
    task = asyncio.create_task(react_agent.run())
    output_queue = react_agent.output_queue
    async def stream_response():
        while not output_queue.empty():
            data = await output_queue.get()
            yield str(data)
    await task
    return StreamingResponse(stream_response(), media_type="text/event-stream")
    
