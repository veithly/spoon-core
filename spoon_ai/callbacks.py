from langchain.callbacks import AsyncIteratorCallbackHandler
from sqlalchemy import select

from api.core.database import get_db
from models.base.model import SessionStateTable


class StorageAsyncIteratorCallbackHandler(AsyncIteratorCallbackHandler):
    human_message: str
    ai_message: str
    session_id: str
    async def on_llm_start(self, serialized, prompts, **kwargs):
        self.human_message = prompts[0][7:]
        return await super().on_llm_start(serialized, prompts, **kwargs)
    async def on_llm_end(self, response, **kwargs):
        self.ai_message = response.generations[0][0].text
        db = get_db()
        session = await db.execute(select(SessionStateTable).where(SessionStateTable.session_id == self.session_id))
        session.chat_history.append({"role": "user", "content": self.human_message})
        session.chat_history.append({"role": "assistant", "content": self.ai_message})
        await db.commit()
        return await super().on_llm_end(response, **kwargs)
