import pytest
from spoon_ai.chat import ChatBot
from examples.custom_agent_example import InfoAssistantAgent
from spoon_ai.agents.base import AgentState

@pytest.fixture
async def agent():
    a = InfoAssistantAgent(llm=ChatBot())
    yield a  # Pass `agent` to the test function
    a.clear()  # Clear state after test
    a.state = AgentState.IDLE  # Reset agent state
    print(f"Agent state reset after test: {a.state}")


@pytest.mark.asyncio
async def test_websearch_agent(agent):
    queries = [
        "Search the current date on the internet.",
        "What is the latest price of Bitcoin?",
        "Explain the specific usage of LangChain.",
        "Get the latest block number from the NEO mainnet."
    ]
    for query in queries:
        print(f"\n=== Query: {query} ===")
        response = await agent.run(query)
        print(f"Response:\n{response}")

        # 验证响应是否有效
        assert response is not None, f"Query `{query}` returned None!"
        assert isinstance(response, str), f"Query `{query}` did not return a string!"
        assert len(response.strip()) > 0, f"Query `{query}` returned an empty string!"

        # 可选：更细致的检查（关键词）
        if "bitcoin" in query.lower():
            assert "price" in response.lower() or "$" in response, "Missing price info in Bitcoin response"
        elif "date" in query.lower():
            assert any(word in response.lower() for word in ["today", "date", "march", "2025"]), "Date not detected in response"
        elif "langchain" in query.lower():
            assert "langchain" in response.lower(), "Missing LangChain explanation"
        elif "block number" in query.lower() or "neo" in query.lower():
            assert any(word in response.lower() for word in ["block", "neo"]), "Missing block info for NEO"
        agent.clear()


@pytest.mark.asyncio
async def test_calculator_agent(agent):
    """
    Test Calculator tool via agent queries with detailed assertions.
    """
    test_cases = [
        {
            "query": "What is 13 factorial?",
            "expected_result": "6227020800"
        },
        {
            "query": "Solve: (5 + 2) * 10 / (3 - 1)",
            "expected_result": "35"
        },
        {
            "query": "What's the square root of 256?",
            "expected_result": "16"
        },
        {
            "query": "How many seconds are there in 3 days?",
            "expected_result": "259200"
        },
        {
            "query": "Convert 100 degrees Celsius to Fahrenheit.",
            "expected_result": "212"
        }
    ]

    for case in test_cases:
        query = case["query"]
        expected = case["expected_result"]

        print(f"\n=== Running Query: {query} ===")
        response = await agent.run(query)
        print(f"Response:\n{response}")

        assert response is not None, f"Query `{query}` returned None"
        assert isinstance(response, str), f"Query `{query}` returned a non-string: {type(response)}"
        assert expected in response.replace(",", ""), f"Expected result '{expected}' not found in response: {response}"

        agent.clear()


@pytest.mark.asyncio
async def test_weatherinfo_agent(agent):
    """
    Test WeatherInfo tool via agent interaction using simulated weather data.
    
    This test checks:
    1. Weather info retrieval for known cities
    2. Error message for unknown cities
    """

    queries = [
        "What's the weather like in Beijing?",
        "Tell me the current weather in Shanghai for the next 3 days.",
        "How's the weather in Guangzhou?",
        "Give me the weather info for Shenzhen today.",
        "Check the weather in Paris."  # Should trigger error
    ]

    expected_assertions = [
        ["Beijing", "Temperature", "Sunny"],         # Simulated Beijing weather
        ["Shanghai", "Condition", "Cloudy"],         # Simulated Shanghai weather
        ["Guangzhou", "Rainy", "Humidity"],          # Simulated Guangzhou weather
        ["Shenzhen", "Overcast", "Temperature"],     # Simulated Shenzhen weather
        ["not found", "Paris"]                       # Expected error for unknown city
    ]

    for query, expected_keywords in zip(queries, expected_assertions):
        print(f"\n=== Query: {query} ===")
        response = await agent.run(query)
        print(f"Response:\n{response}")

        assert response is not None
        assert isinstance(response, str)

        for keyword in expected_keywords:
            assert keyword.lower() in response.lower(), f"Expected '{keyword}' in response but got:\n{response}"

        agent.clear()


@pytest.mark.asyncio
async def test_info_assistant_hybrid_queries(agent):
    """
    Test InfoAssistantAgent handling hybrid queries using simulated (mocked) weather data.
    Only mock-supported cities (Beijing, Shanghai, Guangzhou, Shenzhen) are used for weather.
    """

    test_cases = [
        {
            "query": "What's the weather in Shanghai and how might it influence outdoor activities?",
            "expected_phrases": ["shanghai", "weather", "cloudy"]
        },
        {
            "query": "Get the price of Bitcoin and convert it to RMB.",
            "expected_phrases": ["bitcoin", "price", "rmb", "converted"]
        },
        {
            "query": "Find news about the Ethereum ETF and its market influence.",
            "expected_phrases": ["ethereum", "etf", "news", "impact"]
        },
        {
            "query": "Tell me the weather in Beijing and estimate the temperature difference if it drops by 5 degrees.",
            "expected_phrases": ["beijing", "25", "temperature", "drop", "20"]
        },
        {
            "query": "Who is the leader of China and what's the weather in Guangzhou today?",
            "expected_phrases": ["guangzhou", "rainy", "humidity", "leader", "china"]
        }
    ]

    for case in test_cases:
        query = case["query"]
        print(f"\n=== Running Query: {query} ===")
        response = await agent.run(query)
        print(f"Response:\n{response}")

        assert response is not None, f"Query `{query}` returned None"
        assert isinstance(response, str), f"Query `{query}` returned non-string"

        for phrase in case["expected_phrases"]:
            assert phrase.lower() in response.lower(), f"Expected phrase `{phrase}` not found in response."
