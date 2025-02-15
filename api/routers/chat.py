import asyncio
from typing import AsyncIterable, Awaitable, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.status import HTTP_401_UNAUTHORIZED
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from models.base.chat import CompletionsRequest, Messages, SessionRequest
from models.controller import model_manager
from models.response import OpenAIStreamResponse
from models.workflow import Workflow

from utils import ALGORITHM, SECRET_KEY
import jwt

router = APIRouter(prefix="/v1/chat")
security = HTTPBearer()
def get_current_user(credentials:HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        address = payload.get("sub")
        if not address:
            raise ValueError("Invalid token")
        return address
    except Exception as e:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED)

async def send_message(
    messages_contents: List[Messages],
    session_id: str,
    disconnect_event: asyncio.Event = None,
) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    messages = []
    for message_content in messages_contents:
        if message_content.role == "user":
            messages.append(HumanMessage(message_content.content))
        elif message_content.role == "system":
            messages.append(SystemMessage(message_content.content))
        elif message_content.role == "assistant":
            messages.append(AIMessage(message_content.content))
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid role. Must be one of 'user', 'system', 'assistant'",
            )
        model_id = model_manager.get_model_id_from_session(session_id)
        logger.debug(f"Model ID: {model_id}")
        if model_id.startswith("b'") and model_id.endswith("'"):
            model_id = model_id[2:-1]
        model = model_manager.get_model_via_id(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        workflow = Workflow(model=model, session_id=session_id, callback=callback)

        async def wrap_done(fn: Awaitable, done: asyncio.Event):
            try:
                await fn
            except Exception as e:
                logger.error(e)
            finally:
                done.set()

        task = asyncio.create_task(
            wrap_done(workflow.chains[0].agenerate([messages]), callback.done)
        )
        async for token in callback.aiter():
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
        await task


@router.post("/completions")
async def stream_completions(body: CompletionsRequest):
    logger.debug(f"Received completions request: {body}")
    disconnect_event = asyncio.Event()
    return OpenAIStreamResponse(
        send_message(
            messages_contents=body.messages,
            session_id=body.session_id,
            disconnect_event=disconnect_event,
        ),
        media_type="text/event-stream",
    )

@router.post("/session")
async def create_session(body: SessionRequest, current_address: str = Depends(get_current_user)):
    session_id = model_manager.create_session(body.model_id, current_address)
    return {"data": {"session_id": session_id}, "message": "success"}

@router.get("/session/messages/{id}")
async def get_messsages(id, current_address: str = Depends(get_current_user)):
    models_via_address = model_manager.get_models_via_address(current_address)
    model_ids = [model.id for model in models_via_address]
    model_id = model_manager.get_model_id_from_session(id)
    if model_id not in model_ids:
        raise HTTPException(status_code=401, detail="Unauthorized")
    messages = model_manager.get_chat_history_from_session(id)
    return {"data": {"messages": messages}, "message": "success"}