import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from models.base.model import Model
from models.controller import model_manager
from starlette.status import HTTP_401_UNAUTHORIZED

from utils import  ALGORITHM, SECRET_KEY

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
async def get_model_via_id(id: str, current_address: str = Depends(get_current_user)):
    model = model_manager.get_model_via_id(id)
    if model is None:
        return {"message": "Model not found"}
    return {"data": model.model_dump_json(), "message": "success"}

@router.get("/", tags=["models"])
async def get_models_list(current_address: str = Depends(get_current_user)):
    models = model_manager.get_models_via_address(current_address)
    return {"data": [model.model_dump_json() for model in models], "message": "success"}

@router.post("/", tags=["models"])
async def create_model(model: Model, current_address: str = Depends(get_current_user)):
    model.owner_address = current_address
    model_id = model_manager.save_model(model, current_address)
    return {"data": {"id": model_id}, "message": "success"}


@router.patch("/{id}", tags=["models"])
async def update_model(id: str, model: Model, current_address: str = Depends(get_current_user)):
    model.id = id
    previous_model = model_manager.get_model_via_id(id)
    if current_address != previous_model.owner_address:
        raise HTTPException(status_code=403, detail="You are not allowed to update")
    model_id = model_manager.save_model(model, current_address)
    return {"data": {"id": model_id}, "message": "success"}
