import os
import json
import httpx
import base64
from typing import List, Optional
from urllib.parse import quote

from dotenv import load_dotenv

from .models import (
    Bearer,
    TokenResponse,
    BinaryBearer,
    ContainerPostInfo,
    ContainerInfo,
    ContainerList,
    Eacl,
    SearchRequest,
    ObjectListV2,
    Balance,
    NetworkInfo,
    SuccessResponse,
    UploadAddress,
    ErrorResponse
)
from .utils import generate_simple_signature_params, sign_bearer_token, sign_with_salt


class NeoFSClient:
    def __init__(self, base_url: Optional[str] = None, owner_address: Optional[str] = None, private_key_wif: Optional[str] = None):
        load_dotenv()
        self.base_url = base_url or os.getenv("NEOFS_BASE_URL")
        self.owner_address = owner_address or os.getenv("NEOFS_OWNER_ADDRESS")
        self.private_key_wif = private_key_wif or os.getenv("NEOFS_PRIVATE_KEY_WIF")
        timeout_env = os.getenv("NEOFS_HTTP_TIMEOUT")
        try:
            timeout_value = float(timeout_env) if timeout_env else 30.0
        except (TypeError, ValueError):
            timeout_value = 30.0

        if not all([self.base_url, self.owner_address, self.private_key_wif]):
            raise ValueError("Missing configuration. Provide base_url, owner_address, and private_key_wif or set environment variables.")

        self.http_client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=timeout_value, read=timeout_value, write=timeout_value, pool=None),
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.http_client.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error = ErrorResponse(**error_data)
                raise NeoFSAPIException(error, e.request) from e
            except (json.JSONDecodeError, KeyError):
                raise NeoFSException(f"HTTP error: {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            raise NeoFSException(f"Request error: {e}") from e

    # 1. Auth
    def create_bearer_tokens(self, tokens: List[Bearer], lifetime: int = 100, for_all_users: bool = False) -> List[TokenResponse]:
        headers = {
            'X-Bearer-Owner-Id': self.owner_address,
            'X-Bearer-Lifetime': str(lifetime),
            'X-Bearer-For-All-Users': str(for_all_users).lower()
        }
        content = [t.dict(by_alias=True, exclude_none=True) for t in tokens]
        response = self._request('POST', '/v1/auth', headers=headers, json=content)
        return [TokenResponse(**item) for item in response.json()]

    def get_binary_bearer_token(self) -> BinaryBearer:
        params = generate_simple_signature_params(self.private_key_wif)
        response = self._request('GET', '/v1/auth/bearer', params=params)
        return BinaryBearer(**response.json())

    # 2. Container
    def create_container(
        self,
        container_info: ContainerPostInfo,
        bearer_token: str,
        name_scope_global: bool = True,
        *,
        wallet_connect: bool = False,
    ) -> ContainerInfo:
        signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=wallet_connect)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'X-Bearer-Signature': signature_value,
            'X-Bearer-Signature-Key': signature_key,
        }
        params = {
            'name-scope-global': str(name_scope_global).lower(),
        }
        if wallet_connect:
            params['walletConnect'] = 'true'
        response = self._request('POST', '/v1/containers', headers=headers, params=params, json=container_info.model_dump(by_alias=True))
        
        # The response gives a containerId, but to return a full ContainerInfo, we must fetch it.
        container_id = response.json().get('containerId')
        if not container_id:
            raise NeoFSException("Container creation did not return a container ID.")
        return self.get_container(container_id)


    def get_container(self, container_id: str) -> ContainerInfo:
        response = self._request('GET', f'/v1/containers/{container_id}')
        return ContainerInfo(**response.json())

    def delete_container(self, container_id: str, *, bearer_token: str | None = None, wallet_connect: bool = False) -> SuccessResponse:
        if bearer_token:
            signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=wallet_connect)
            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'X-Bearer-Signature': signature_value,
                'X-Bearer-Signature-Key': signature_key,
            }
            params = {'walletConnect': 'true'} if wallet_connect else None
            response = self._request('DELETE', f'/v1/containers/{container_id}', headers=headers, params=params)
        else:
            params = generate_simple_signature_params(self.private_key_wif)
            response = self._request('DELETE', f'/v1/containers/{container_id}', params=params)
        return SuccessResponse(**response.json())

    def get_container_eacl(self, container_id: str) -> Eacl:
        response = self._request('GET', f'/v1/containers/{container_id}/eacl')
        return Eacl(**response.json())

    def set_container_eacl(self, container_id: str, eacl: Eacl) -> SuccessResponse:
        params = generate_simple_signature_params(self.private_key_wif)
        response = self._request('PUT', f'/v1/containers/{container_id}/eacl', params=params, json=eacl.model_dump(by_alias=True, exclude={'containerId'}))
        return SuccessResponse(**response.json())

    # 3. List Containers
    def list_containers(self, owner_id: str, offset: int | None = None, limit: int | None = None) -> ContainerList:
        params = {'ownerId': owner_id}
        if offset is not None:
            params['offset'] = offset
        if limit is not None:
            params['limit'] = limit
        response = self._request('GET', '/v1/containers', params=params)
        return ContainerList(**response.json())

    # 4. Object Upload
    def upload_object(
        self,
        container_id: str,
        bearer_token: str,
        content: bytes,
        *,
        attributes: Optional[dict] = None,
        expiration_rfc3339: Optional[str] = None,
        expiration_duration: Optional[str] = None,
        expiration_timestamp: Optional[int] = None,
        timeout: Optional[float] = 180,
    ) -> UploadAddress:
        signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=False)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/octet-stream',
            'X-Bearer-Signature': signature_value,
            'X-Bearer-Signature-Key': signature_key,
        }
        if attributes:
            headers['X-Attributes'] = json.dumps(attributes)
        if expiration_rfc3339:
            headers['X-Neofs-Expiration-RFC3339'] = expiration_rfc3339
        if expiration_duration:
            headers['X-Neofs-Expiration-Duration'] = expiration_duration
        if expiration_timestamp is not None:
            headers['X-Neofs-Expiration-Timestamp'] = str(expiration_timestamp)
            
        params = generate_simple_signature_params(self.private_key_wif, payload_parts=(headers['X-Bearer-Signature'].encode(),))

        request_kwargs = {'headers': headers, 'content': content, 'params': params}
        if timeout is not None:
            request_kwargs['timeout'] = timeout
        response = self._request('POST', f'/v1/objects/{container_id}', **request_kwargs)
        return UploadAddress(**response.json())

    # 5. Object Download & Retrieval
    def download_object_by_id(
        self,
        container_id: str,
        object_id: str,
        bearer_token: str,
        *,
        download: bool | None = None,
        range_header: str | None = None,
    ) -> httpx.Response:
        signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=False)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'X-Bearer-Signature': signature_value,
            'X-Bearer-Signature-Key': signature_key,
        }
        if range_header:
            headers['Range'] = range_header
        params = generate_simple_signature_params(self.private_key_wif, payload_parts=(signature_value.encode(),))
        if download is not None:
            params['download'] = str(download).lower()
        return self._request('GET', f'/v1/objects/{container_id}/by_id/{object_id}', headers=headers, params=params)

    def get_object_header_by_id(
        self,
        container_id: str,
        object_id: str,
        bearer_token: str,
        *,
        range_header: str | None = None,
    ) -> httpx.Response:
        signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=False)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'X-Bearer-Signature': signature_value,
            'X-Bearer-Signature-Key': signature_key,
        }
        if range_header:
            headers['Range'] = range_header
        params = generate_simple_signature_params(self.private_key_wif, payload_parts=(signature_value.encode(),))
        return self._request('HEAD', f'/v1/objects/{container_id}/by_id/{object_id}', headers=headers, params=params)

    def download_object_by_attribute(
        self,
        container_id: str,
        attr_key: str,
        attr_val: str,
        bearer_token: str,
        *,
        download: bool | None = None,
        range_header: str | None = None,
    ) -> httpx.Response:
        signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=False)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'X-Bearer-Signature': signature_value,
            'X-Bearer-Signature-Key': signature_key,
        }
        if range_header:
            headers['Range'] = range_header
        encoded_key = quote(attr_key, safe="")
        encoded_val = quote(attr_val, safe="")
        params = generate_simple_signature_params(self.private_key_wif, payload_parts=(signature_value.encode(),))
        if download is not None:
            params['download'] = str(download).lower()
        return self._request('GET', f'/v1/objects/{container_id}/by_attribute/{encoded_key}/{encoded_val}', headers=headers, params=params)
        
    def get_object_header_by_attribute(
        self,
        container_id: str,
        attr_key: str,
        attr_val: str,
        bearer_token: str,
        *,
        range_header: str | None = None,
    ) -> httpx.Response:
        signature_value, signature_key = sign_bearer_token(bearer_token, self.private_key_wif, wallet_connect=False)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'X-Bearer-Signature': signature_value,
            'X-Bearer-Signature-Key': signature_key,
        }
        if range_header:
            headers['Range'] = range_header
        encoded_key = quote(attr_key, safe="")
        encoded_val = quote(attr_val, safe="")
        params = generate_simple_signature_params(self.private_key_wif, payload_parts=(signature_value.encode(),))
        return self._request('HEAD', f'/v1/objects/{container_id}/by_attribute/{encoded_key}/{encoded_val}', headers=headers, params=params)
    
    def delete_object(self, container_id: str, object_id: str) -> SuccessResponse:
        params = generate_simple_signature_params(self.private_key_wif)
        response = self._request('DELETE', f'/v1/objects/{container_id}/{object_id}', params=params)
        return SuccessResponse(**response.json())

    # 6. Object Search
    def search_objects(
        self,
        container_id: str,
        bearer_token: str,
        search_request: SearchRequest,
        *,
        cursor: str = "",
        limit: int = 100,
    ) -> ObjectListV2:
        payload_to_sign = base64.b64decode(bearer_token)
        signature_components = sign_with_salt(self.private_key_wif, payload_to_sign)
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json',
            'X-Bearer-Owner-Id': self.owner_address,
            'X-Bearer-Signature': signature_components.signature_header(),
            'X-Bearer-Signature-Key': signature_components.public_key,
        }
        params = {
            'cursor': cursor,
            'limit': limit,
            **generate_simple_signature_params(components=signature_components),
        }
        response = self._request('POST', f'/v2/objects/{container_id}/search', headers=headers, params=params, json=search_request.model_dump(by_alias=True))
        return ObjectListV2(**response.json())

    # 7. Accounting
    def get_balance(self, address: Optional[str] = None) -> Balance:
        address_to_query = address or self.owner_address
        response = self._request('GET', f'/v1/accounting/balance/{address_to_query}')
        return Balance(**response.json())

    # 8. Network
    def get_network_info(self) -> NetworkInfo:
        response = self._request('GET', '/v1/network-info')
        return NetworkInfo(**response.json())


class NeoFSException(Exception):
    """Base exception for the NeoFS client."""
    pass

class NeoFSAPIException(NeoFSException):
    """Raised when the API returns an error."""
    def __init__(self, error: ErrorResponse, request: httpx.Request):
        self.error = error
        self.request = request
        super().__init__(f"API Error {error.code} ({error.type}): {error.message}")
        