import uuid

from models.base.model import Model, ModelTable, SessionStateTable
from sqlalchemy.orm import Session

from utils import REDIS_HOST, REDIS_PASSWORD, REDIS_PORT

from .base import BaseManager


class ModelManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.table = self.get_table("models")
        self.db = Session(bind=self.engine)

    def save_model(self, model: Model, user_address: str):
        model.id = uuid.uuid4().hex
        model_table = ModelTable(id=model.id, chains=model.chains, owner_address=user_address)
        self.db.add(model_table)
        self.db.commit()
        return model.id
        

    def get_model_via_id(self, model_id: str) -> Model:
        model_table =  self.db.query(ModelTable).filter(ModelTable.id == model_id).first()
        if not model_table:
            return None
        return Model.model_validate_json(model_table.model_dump_json())
    
    def get_models_via_address(self, user_address: str) -> list[Model]:
        model_tables = self.db.query(ModelTable).filter(ModelTable.owner_address == user_address).all()
        return [Model.model_validate_json(model_table.model_dump_json()) for model_table in model_tables]

    def create_session(self, model_id: str, owner_address: str) -> str:
        model = self.get_model_via_id(model_id)
        if model.owner_address != owner_address:
            raise ValueError("You are not allowed to access this model")
        session_id = uuid.uuid4().hex
        session_table = SessionStateTable(session_id=session_id, model_id=model_id)
        self.db.add(session_table)
        self.db.commit()
        return session_id

    def get_model_id_from_session(self, session_id: str) -> str:
        session_table = self.db.query(SessionStateTable).filter(SessionStateTable.session_id == session_id).first()
        return session_table.model_id
    
    def get_chat_history_from_session(self, session_id: str) -> list:
        session_table = self.db.query(SessionStateTable).filter(SessionStateTable.session_id == session_id).first()
        return session_table.chat_history


model_manager = ModelManager()
