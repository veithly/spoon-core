import asyncio
import uuid
from logging import getLogger
from typing import AsyncIterable, Awaitable, List

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_401_UNAUTHORIZED

from api.core.database import get_db
from models.base.chat import CompletionsRequest, Messages, SessionRequest
from models.base.model import ModelTable, SessionStateTable
from models.response import OpenAIStreamResponse
from models.workflow import Workflow
from spoon_ai.callbacks import StorageAsyncIteratorCallbackHandler
from utils import ALGORITHM, SECRET_KEY

logger = getLogger("api")

router = APIRouter(prefix="/v1/chat")
security = HTTPBearer()
def get_current_user(credentials:HTTPAuthorizationCredentials = Depends(security), db: AsyncSession = Depends(get_db)):
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
    db: AsyncSession = Depends(get_db)
) -> AsyncIterable[str]:
    callback = StorageAsyncIteratorCallbackHandler(session_id=session_id)
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
        model = await db.execute(select(ModelTable).where(ModelTable.id == session_id))
        logger.debug(f"Model ID: {model.id}")
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
async def create_session(body: SessionRequest, current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    session_id = uuid.uuid4().hex
    session_table = SessionStateTable(id=session_id, model_id=body.model_id, owner_address=current_address)
    db.add(session_table)
    return {"data": {"session_id": session_id}, "message": "success"}

@router.get("/session/messages/{id}")
async def get_messsages(id, current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    session = await db.execute(select(SessionStateTable).where(SessionStateTable.id == id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.owner_address != current_address:
        raise HTTPException(status_code=401, detail="Unauthorized")
    messages = session.chat_history
    return {"data": {"messages": messages}, "message": "success"}

@router.get("/session/ids/{model_id}")
async def get_session_ids_via_model_id(model_id: str, current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sessions = await db.execute(select(SessionStateTable).where(
        (SessionStateTable.model_id == model_id) & 
        (SessionStateTable.owner_address == current_address)
    ))
    return {"data": {"sessions": [session.id for session in sessions]}, "message": "success"}
