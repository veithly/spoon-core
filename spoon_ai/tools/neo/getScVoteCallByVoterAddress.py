from .tool_collection import getScVoteCallByVoterAddress
from spoon_ai.tools.base import BaseTool

class GetScVoteCallByVoterAddress(BaseTool):
    name: str = "get_sc_vote_call_by_voter_address"
    description: str = "Get the sc vote call by voter address"
    parameters: dict = {
        "typpe": "object",
        "properties": {
            "voter_address": {
                "type": "string",
                "description": "The address of the voter"
            }
        },
        "required": ["voter_address"]
    }
    
    async def execute(self, voter_address: str) -> str:
        return getScVoteCallByVoterAddress(voter_address)
    
    
    