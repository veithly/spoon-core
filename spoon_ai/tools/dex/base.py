import json
import os

import requests
from pydantic import Field

from spoon_ai.tools.base import BaseTool

BITQUERY_API_KEY = os.getenv("BITQUERY_API_KEY")
BITQUERY_CLIENT_ID = os.getenv("BITQUERY_CLIENT_ID")
BITQUERY_CLIENT_SECRET = os.getenv("BITQUERY_CLIENT_SECRET")

class DexBaseTool(BaseTool):
    bitquery_endpoint: str = "https://streaming.bitquery.io/graphql"
    name: str = Field(description="The name of the tool")
    description: str = Field(description="A description of the tool")
    parameters: dict = Field(description="The parameters of the tool")
    graph_template: str = Field(description="The graph template of the tool")

    def oAuth(self):
        url = "https://oauth2.bitquery.io/oauth2/token"
        payload = f'grant_type=client_credentials&client_id={BITQUERY_CLIENT_ID}&client_secret={BITQUERY_CLIENT_SECRET}&scope=api'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.request("POST", url, headers=headers, data=payload)
        resp = json.loads(response.text)
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resp['access_token']}"
        }

    async def execute(self, **kwargs) -> str:
        graph = self.graph_template.format(**kwargs)
        response = requests.post(self.bitquery_endpoint, json={"query": graph}, headers=self.oAuth())
        if response.status_code != 200:
            raise Exception(f"Failed to execute tool: {response.text}")
        return response.json()
        