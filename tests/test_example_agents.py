import pytest
from spoon_ai.chat import ChatBot
from examples.custom_agent_example import InfoAssistantAgent
from spoon_ai.agents.base import AgentState
from asserts import AgentTestAssertions


@pytest.fixture
async def agent():
    a = InfoAssistantAgent(llm=ChatBot())
    yield a  # Pass `agent` to the test function
    a.clear()  # Clear state after test
    a.state = AgentState.IDLE  # Reset agent state
    print(f"Agent state reset after test: {a.state}")

@pytest.mark.asyncio
@pytest.mark.parametrize("query, keywords", [
    ("Search the current date on the internet.", ["today", "date", "march", "2025"]),
    ("What is the latest price of Bitcoin?", ["price", "$"]),
    ("Explain the specific usage of LangChain.", ["langchain"]),
    ("Get the latest block number from the NEO mainnet.", ["block", "neo"]),
])
async def test_websearch_agent(agent, query, keywords):
    print(f"=== Query: {query} ===")
    response = await agent.run(query)
    print(f"Response:{response}")

    AgentTestAssertions.assert_keywords_in_response(response,query,keywords)



@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected", [
    ("What is 13 factorial?", "6227020800"),
    ("Solve: (5 + 2) * 10 / (3 - 1)", "35"),
    ("What's the square root of 256?", "16"),
    ("How many seconds are there in 3 days?", "259200"),
    ("Convert 100 degrees Celsius to Fahrenheit.", "212")
])
async def test_calculator_agent(agent, query, expected):
    """
    Parametrized test: Calculator tool via agent queries with detailed assertions.
    """
    print(f"=== Running Query: {query} ===")
    response = await agent.run(query)
    print(f"Response:{response}")

    AgentTestAssertions.assert_response_matches_exact(response,query,expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected_keywords", [
    (
        "What's the weather like in Beijing?",
        ["Beijing", "Temperature", "Sunny"]
    ),
    (
        "Tell me the current weather in Shanghai for the next 3 days.",
        ["Shanghai", "Condition", "Cloudy"]
    ),
    (
        "How's the weather in Guangzhou?",
        ["Guangzhou", "Rainy", "Humidity"]
    ),
    (
        "Give me the weather info for Shenzhen today.",
        ["Shenzhen", "Overcast", "Temperature"]
    ),
    (
        "Check the weather in Paris.",
        ["not found", "Paris"]
    )
])
async def test_weatherinfo_agent(agent, query, expected_keywords):
    """
    Parametrized test: WeatherInfo tool via agent interaction using simulated weather data.

    This test checks:
    1. Weather info retrieval for known cities
    2. Error message for unknown cities
    """
    print(f"=== Query: {query} ===")
    response = await agent.run(query)
    print(f"Response:{response}")

    AgentTestAssertions.assert_keywords_in_response(response,query,expected_keywords)


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected_phrases", [
    (
        "What's the weather in Shanghai and how might it influence outdoor activities?",
        ["shanghai", "weather", "cloudy"]
    ),
    (
        "Get the price of Bitcoin and convert it to RMB.",
        ["bitcoin"]
    ),
    (
        "Find news about the Ethereum ETF and its market influence.",
        ["ethereum", "etf", "news", ]
    ),
    (
        "Tell me the weather in Beijing and estimate the temperature difference if it drops by 5 degrees.",
        ["beijing", "25", "temperature", "drop", "20"]
    ),
    (
        "Who is the leader of China and what's the weather in Guangzhou today?",
        ["guangzhou", "rainy", "leader", "china"]
    )
])
async def test_info_assistant_hybrid_queries(agent, query, expected_keywords):
    """
    Parametrized test: InfoAssistantAgent hybrid queries using simulated (mocked) weather data.
    """

    print(f"=== Running Query: {query} ===")
    response = await agent.run(query)
    print(f"Response:{response}")

    AgentTestAssertions.assert_keywords_in_response(response,query,expected_keywords)

