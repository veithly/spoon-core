import httpx
import time
from hashlib import sha1
from spoon_ai.tools.gopluslabs.env import GO_PLUS_LABS_APP_KEY, GO_PLUS_LABS_APP_SECRET

def get_sign_and_time(app_key: str = '', app_secret: str = '', t: int | str = 0) -> (str, str):
    """
    "sign" is utilized to get a gopluslabs API access token
    https://docs.gopluslabs.io/reference/getaccesstokenusingpost
    Concatenate app_key, time, app_secret in turn, and do sha1().
    Example
    app_key = mBOMg20QW11BbtyH4Zh0
    time = 1647847498   # in seconds
    app_secret = V6aRfxlPJwN3ViJSIFSCdxPvneajuJsh
    sign = sha1(mBOMg20QW11BbtyH4Zh01647847498V6aRfxlPJwN3ViJSIFSCdxPvneajuJsh)
    = 7293d385b9225b3c3f232b76ba97255d0e21063e
    """
    app_key = app_key or GO_PLUS_LABS_APP_KEY
    app_secret = app_secret or GO_PLUS_LABS_APP_SECRET
    t = str(t) if t else str(int(time.time()))
    concatenated: str = app_key + t + app_secret
    sign = sha1(concatenated.encode(encoding="utf-8")).hexdigest()
    return sign, t

def test_get_sign_and_time():
    sign, time = get_sign_and_time('mBOMg20QW11BbtyH4Zh0', 'V6aRfxlPJwN3ViJSIFSCdxPvneajuJsh', 1647847498)
    assert sign == "7293d385b9225b3c3f232b76ba97255d0e21063e"

def get_token() -> str:
    """
    returns: Bearer aaaaauth_token...
    """
    sign, t = get_sign_and_time()
    r = httpx.post('https://api.gopluslabs.io/api/v1/token', json={
        "app_key": GO_PLUS_LABS_APP_KEY,
        "sign": sign,
        "time": t
    })
    r = r.json()
    expire_time = r["result"]["expires_in"]
    return r["result"]["access_token"]

GO_PLUS_LABS_AUTH_TOKEN = get_token()  # with "Bearer " prefix

async def raise_on_4xx_5xx(response):
    await response.aread()
    response.raise_for_status()


go_plus_labs_client_v1 = httpx.AsyncClient(
    base_url='https://api.gopluslabs.io/api/v1'.removesuffix('/'),
    headers={'Authorization': GO_PLUS_LABS_AUTH_TOKEN},
    event_hooks={'response': [raise_on_4xx_5xx]},
)

go_plus_labs_client_v2 = httpx.AsyncClient(
    base_url='https://api.gopluslabs.io/api/v2'.removesuffix('/'),
    headers={'Authorization': GO_PLUS_LABS_AUTH_TOKEN},
    event_hooks={'response': [raise_on_4xx_5xx]},
)