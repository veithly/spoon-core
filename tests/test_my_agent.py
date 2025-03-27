import pytest
import re
from my_agent import MyInfoAgent
from spoon_ai.chat import ChatBot
from spoon_ai.agents.base import AgentState
from asserts import AgentTestAssertions


# FIXTURE: Create Agent and reset it after each test
@pytest.fixture
async def agent():
    """Fixture to create an agent and ensure it resets state after each test."""
    a = MyInfoAgent(llm=ChatBot())
    yield a  # Pass `agent` to the test function
    a.clear()  # Clear state after test
    a.state = AgentState.IDLE  # Reset agent state
    print(f"Agent state reset after test: {a.state}")


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected_phrases", [
    # 1. GitHub commit query
    (
        "How many commits in neo-project/neo master branch this month?",
        ["total commits", "neo-project/neo", "master"]
    ),

    # 2. Currency conversion
    (
        "Convert 100 USD to CNY",
        ["usd", "cny", "100", "=", "rate"]
    ),

    # 3. Weather + Clothing Tips
    (
        "What's the weather and pollution in changsha today?",
        ["changsha", "temperature", "pm2.5", "clothing", "air quality", "👕", "😷"]
    ),

    # 4. Combined query: weather + exchange rate
    (
        "Tell me the weather in Shanghai and convert 50 EUR to JPY",
        ["shanghai", "temperature", "eur", "jpy", "rate"]
    ),

    # 5. Wrong city test (fallback behavior of weather tool)
    (
        "What's the weather in Atlantis?",
        ["atlantis", "mythical", "does not exist", "cannot provide weather informatio"]
    ),

    # 6. Wrong GitHub storehouse (fallback behavior of GitHubCommitStatsTool )
    (
        "How many commits in neo-project/aaa master branch this month?",
        ["does not exist"]
    ),

    # 7. Combo: weather + exchange rate + GitHub commit
    (
        "Give me the weather in shaoyang, convert 10 GBP to USD, and check this month's commits for neo-project/neo-devpack-dotnet",
        ["shaoyang", "temperature", "pm2.5", "gbp", "usd", "rate", "neo-project/neo-devpack-dotnet", "total commits"]
    ),

    # 8. Combo: invalid city + valid exchange rate
    (
        "What is the weather in MiddleEarth and convert 200 EUR to CNY",
        ["eur", "cny", "rate", "200"]
    ),

    # 9. Combo: invalid city + invalid exchange rate
    (
        "What is the weather in Neverland and convert 500 ABC to XYZ",
        [ "provide valid city names and currency codes" ]
    )

    # 10. Simple query
    (
        "xinshao weather",
        ["xinshao", "temperature", "pm2.5", "clothing", "air quality", "👕", "😷"]
    ),

    # 11. Simple query of GitHub commits provides inaccurate information
    (
        "Hneo-projects/proposals  commit  3 month?",
        []
    ),


    # 12 Other unrelated tool queries
    (
        "Get the latest price of neo on the blockchain",
        []
    ),



])
async def test_my_info_agent_queries(agent, query, expected_phrases):
    print(f"\n--- Running Query: {query} ---")
    response = await agent.run(query)
    print(f"Response:\n{response}")

    AgentTestAssertions.assert_keywords_in_response(response,query,expected_phrases)