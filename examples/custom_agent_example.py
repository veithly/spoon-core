#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom Agent Example

This example demonstrates how to create custom tools and Agents, and how to run them.
"""

import asyncio
import os
from typing import List, Optional

from pydantic import Field

from spoon_ai.agents import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager, Terminate
from spoon_ai.tools.base import BaseTool, ToolResult


# 1. Create custom tools
class WebSearch(BaseTool):
    """Web search tool"""
    name: str = "web_search"
    description: str = "Search the internet for information. Use this tool when you need to find the latest information or facts."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return"
            }
        },
        "required": ["query"]
    }

    async def execute(self, query: str, max_results: int = 5) -> str:
        """Execute web search"""
        # These are simulated search results, in a real application you would call an actual search API
        results = [
            {"title": "Search Result 1", "snippet": f"This is the first search result about {query}"},
            {"title": "Search Result 2", "snippet": f"This is another relevant information about {query}"},
            {"title": "Search Result 3", "snippet": f"Here is more detailed content about {query}"},
            {"title": "Search Result 4", "snippet": f"This is the historical background of {query}"},
            {"title": "Search Result 5", "snippet": f"Latest developments regarding {query}"}
        ]
        
        # Limit the number of results
        results = results[:max_results]
        
        # Format the results
        formatted_results = "\n\n".join([
            f"Title: {result['title']}\nSummary: {result['snippet']}"
            for result in results
        ])
        
        return f"Results for search query '{query}':\n\n{formatted_results}"


class Calculator(BaseTool):
    """Calculator tool"""
    name: str = "calculator"
    description: str = "Perform mathematical calculations. Use this tool when you need to calculate mathematical expressions."
    parameters: dict = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to calculate"
            }
        },
        "required": ["expression"]
    }

    async def execute(self, expression: str) -> str:
        """Perform mathematical calculation"""
        try:
            # Warning: In a real application, you should use a safer way to perform calculations
            # eval can pose security risks, this is just for demonstration
            result = eval(expression)
            return f"The result of expression '{expression}' is: {result}"
        except Exception as e:
            return f"Error calculating expression '{expression}': {str(e)}"


class WeatherInfo(BaseTool):
    """Weather information tool"""
    name: str = "weather_info"
    description: str = "Get weather information for a specified city. Use this tool when the user asks about weather conditions."
    parameters: dict = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            },
            "days": {
                "type": "integer",
                "description": "Number of forecast days (1-7)"
            }
        },
        "required": ["city"]
    }

    async def execute(self, city: str, days: int = 1) -> str:
        """Get weather information"""
        # This is simulated weather data, in a real application you would call a weather API
        weather_data = {
            "Beijing": {"temperature": "25°C", "condition": "Sunny", "humidity": "40%"},
            "Shanghai": {"temperature": "28°C", "condition": "Cloudy", "humidity": "65%"},
            "Guangzhou": {"temperature": "32°C", "condition": "Rainy", "humidity": "80%"},
            "Shenzhen": {"temperature": "30°C", "condition": "Overcast", "humidity": "75%"}
        }
        
        if city in weather_data:
            data = weather_data[city]
            return f"Weather information for {city}:\nTemperature: {data['temperature']}\nCondition: {data['condition']}\nHumidity: {data['humidity']}\nForecast days: {days}"
        else:
            return f"Sorry, weather information for {city} not found."


# 2. Create a custom Agent class
class InfoAssistantAgent(ToolCallAgent):
    """Information Assistant Agent"""
    name: str = "info_assistant"
    description: str = "An assistant that can search for information, perform calculations, and check weather"
    
    system_prompt: str = """You are an information assistant that can help users find information, perform calculations, and check weather.
    You can use the following tools to complete tasks:
    1. web_search - Search the internet for information
    2. calculator - Perform mathematical calculations
    3. weather_info - Get weather information for a specified city
    
    Please choose the appropriate tool based on the user's question and provide useful answers.
    If the user's question doesn't require using a tool, answer directly.
    """
    
    next_step_prompt: str = "What should be the next step?"
    
    max_steps: int = 10
    
    # Define available tools
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        WebSearch(),
        Calculator(),
        WeatherInfo(),
        Terminate()
    ]))


# 3. Create a custom Agent directly using ToolCallAgent
async def create_custom_agent_directly():
    """Create a custom Agent directly using ToolCallAgent"""
    # Create a tool manager
    tool_manager = ToolManager([
        WebSearch(),
        Calculator(),
        WeatherInfo()
    ])
    
    # Create an Agent
    custom_agent = ToolCallAgent(
        name="direct_custom_agent",
        description="Directly created custom Agent",
        llm=ChatBot(),  # Use the default LLM
        avaliable_tools=tool_manager,
        system_prompt="""You are a multi-functional assistant that can search for information, perform calculations, and check weather.
        Please choose the appropriate tool based on the user's question and provide useful answers.""",
        max_steps=10
    )
    
    return custom_agent


# 4. Run the example
async def main():
    # Ensure necessary API keys are set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables not set")
        print("Please set at least one API key to use LLM functionality")
        return
    
    # Create an InfoAssistantAgent instance
    info_agent = InfoAssistantAgent(llm=ChatBot())
    
    # Run the Agent
    print("=== Using InfoAssistantAgent ===")
    response = await info_agent.run("What's the weather like in Beijing today?")
    print(f"Answer: {response}\n")
    
    # Reset the Agent state
    info_agent.clear()
    
    # Run the Agent again with a different question
    response = await info_agent.run("Calculate the result of (15 * 7) + 22")
    print(f"Answer: {response}\n")
    
    # Create a custom Agent directly
    direct_agent = await create_custom_agent_directly()
    
    # Run the directly created Agent
    print("=== Using directly created Agent ===")
    response = await direct_agent.run("Search for the latest advances in artificial intelligence")
    print(f"Answer: {response}\n")


if __name__ == "__main__":
    asyncio.run(main()) 