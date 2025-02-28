import uuid

import jwt
from api.core.database import get_db
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from models.base.model import Model, ModelTable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_401_UNAUTHORIZED

from utils import ALGORITHM, SECRET_KEY

router = APIRouter(prefix="/v1/models")
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

@router.get("/{id}", tags=["models"])
async def get_model_via_id(id: str, current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    model = await db.execute(select(ModelTable).where(ModelTable.id == id))
    if model is None:
        return {"message": "Model not found"}
    return {"data": model.model_dump_json(), "message": "success"}

@router.get("/", tags=["models"])
async def get_models_list(current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)  ):
    models = await db.execute(select(ModelTable).where(ModelTable.owner_address == current_address))
    return {"data": [model.model_dump_json() for model in models], "message": "success"}

@router.post("/", tags=["models"])
async def create_model(model: Model, current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    model.owner_address = current_address
    model_id = uuid.uuid4().hex
    model_table = ModelTable(id=model_id, chains=model.chains, owner_address=current_address)
    db.add(model_table)
    return {"data": {"id": model_id}, "message": "success"}


@router.patch("/{id}", tags=["models"])
async def update_model(id: str, model: Model, current_address: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    model.id = id
    previous_model = await db.execute(select(ModelTable).where(ModelTable.id == id))
    if current_address != previous_model.owner_address:
        raise HTTPException(status_code=403, detail="You are not allowed to update")
    model_id = previous_model.id
    model_table = ModelTable(id=model_id, chains=model.chains, owner_address=current_address)
    db.add(model_table)
    return {"data": {"id": model_id}, "message": "success"}
