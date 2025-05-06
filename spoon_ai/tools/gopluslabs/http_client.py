import httpx
from env import GO_PLUS_LABS_AUTH_TOKEN

def raise_on_4xx_5xx(response):
    response.read()
    response.raise_for_status()


go_plus_labs_client = httpx.AsyncClient(
    base_url='https://api.gopluslabs.io/api/v1'.removesuffix('/'),
    headers={'Authorization': f'Bearer {GO_PLUS_LABS_AUTH_TOKEN}'},
    event_hooks={'response': [raise_on_4xx_5xx]},
)
