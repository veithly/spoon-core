# How to Build a SmartWeatherTool Agent with SpoonAI

In this example, you’ll learn how to build an AI Agent based on the ReAct (Reasoning + Acting) paradigm using SpoonAI’s `ToolCallAgent` framework. This agent is capable of iterative reasoning and decision-making through tool invocation.

You'll define your own tools, configure their behavior, and construct a fully functioning ReAct-style agent — all in Python, with no extra infrastructure required. The code can be executed locally in your IDE or notebook environment.

### What You’ll Build

In this example, you’ll build a smart Agent that helps users check weather and GitHub commit activity

The Agent is equipped with two powerful tools:

- smart_weather(city) — Retrieves the current temperature for a given city. Based on this data, it offers outfit suggestions
- github_commits(repo, branch) — Returns the total number of commits made to a specified GitHub repository branch during the current month.

Just like all SpoonAI tools, the Agent itself does not directly execute these functions. Instead, it interprets the user’s intent, selects the appropriate tool, and delegates the execution to your Python backend. The results are then passed back to the Agent for reasoning.

After receiving tool outputs, the Agent analyzes the data and generates helpful, context-aware responses. For example:

- When asked about the weather, it might recommend outfit suggestions

- When asked about GitHub contributions, it can summarize commit activity for a given project and branch.

This approach allows your Agent to go beyond simple Q&A—making it an intelligent assistant capable of interacting with real-time data and providing actionable insights.

### How It Works

The agent follows this loop:

- User inputs a natural language query

- Agent uses system prompt to understand the task

- Agent selects the most relevant tool (SmartWeatherTool or GitHubCommitStatsTool)

- Agent passes inputs, tool executes real-time API request

- Agent receives tool results and continues reasoning

- A Final response is generated and returned

### 1、Define Your Tools

SpoonAI Agents rely on tools to interact with external data sources and perform real-world actions. In this example, we define two tools:

- SmartWeatherTool: provides real-time weather
- GitHubCommitStatsTool: retrieves commit activity from a GitHub repository.

#### 1.1 The following are custom tool templates

First, you need to create custom tools.

In order for SpoonAI to understand the purpose of these functions, we need to describe them using a specific pattern. We will create a tool class, each containing three properties (name, description, and parameters), and an asynchronous method execute. Each tool class should inherit from the `BaseTool` class:

Each tool is a Python class that inherits from BaseTool, and must define:

```python
from spoon_ai.tools.base import BaseTool


class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "This is a custom tool for performing specific tasks"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of the first parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Description of the second parameter"
            }
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str, param2: int = 0) -> str:
        """Implement the tool's specific logic"""
        # Implement your tool logic here
        result = f"Processing parameters: {param1}, {param2}"
        return result
```

Each tool is a Python class that inherits from BaseTool, and must define:

- `name`: The unique name of the tool
- `description`: A detailed description of the tool (AI will decide when to use it based on this)
- `parameters`: JSON Schema definition of the tool parameters
- `execute()`: Method implementing the tool's specific logic

#### 1.2 Use the above tool template to create our tool - GitHubCommitStatsTool

GitHubCommitStatsTool is a custom tool that fetches the number of commits made to a specific GitHub repository branch within the current month. This tool is particularly useful for generating monthly contribution stats.

Each tool must define the following components:

- `name`: "github_commits"
- `description`: "Get number of commits for a GitHub repo in a date range."
- `parameters`: "repo,branch"
- `execute()`: "Request the https://api.github.com/repo/name of a git repository, get the number of commits and then summarize them"

🧱 Tool Definition

```python
class GitHubCommitStatsTool(BaseTool):
    name: str = "github_commits"
    description: str = "Get number of commits for a GitHub repo in a date range."
    parameters: dict = {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "e.g., 'neo-project/neo'"},
            "branch": {"type": "string", "description": "Branch name, e.g., 'master'"}
        },
        "required": ["repo", "branch"]
    }
```

⚙️ Implementation of execute
This method sends a request to GitHub’s REST API to retrieve commits from a repo and counts them.
1、Get data for this request https://api.github.com/repos/{repo}/commits?sha={branch}&time
2、Statistics on commits in returned data

```python
  async def execute(self, repo: str, branch: str) -> str:
        # Define the time range of the data
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(day=1).isoformat()
        until = now.isoformat()

        commits = []
        page = 1


        async with aiohttp.ClientSession() as session:
            while True:
                url = f"https://api.github.com/repos/{repo}/commits?sha={branch}&since={start_of_month}&until={until}&per_page=100&page={page}"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return f"Failed to fetch commits: {resp.status}"
                    data = await resp.json()
                    if not data:
                        break
                    commits.extend(data)
                    page += 1

        return f"Total commits in {repo} ({branch}) since {start_of_month[:10]}: {len(commits)}"
```

#### 1.3 Use the above tool template to create our tool - SmartWeatherTool

Each tool is a Python class that inherits from BaseTool, and must define:

- `name`: "smart_weather"
- `description`: "Get weather, outfit suggestions, and air quality analysis for a city."
- `parameters`: city
- `execute()`: "The logic for fetching and returning the result"

```python
class SmartWeatherTool(BaseTool):
    """Smart Weather Tool"""
  name: str = "smart_weather"
  description: str = "Get weather and outfit suggestions for a city."
  parameters: dict = {
      "type": "object",
      "properties": {
          "city": {
              "type": "string",
              "description": "City name, e.g., 'Beijing'"
          }
      },
      "required": ["city"]
  }

```

Next, we will implement what we want to do in execute. What request do you want to construct and how to handle the return value?

1. Get longitude and latitude location information
2. Get weather information based on longitude and latitude, and make clothing recommendations based on the weather

````python
from datetime import datetime, timezone
from pydantic import Field
import aiohttp
import asyncio

    async def execute(self, city: str) -> str:
        # Step 1: Get latitude & longitude
        geocode_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"

        async with aiohttp.ClientSession() as session:
            async with session.get(geocode_url) as resp:
                if resp.status != 200:
                    return f"Failed to obtain the geographic location, status code:{resp.status}"
                geocode_data = await resp.json()

        if not geocode_data:
            return f"Unable to find geographic information for city {city}"

        lat = geocode_data[0]["lat"]
        lon = geocode_data[0]["lon"]

        # Step 2: Get weather info
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(weather_url) as resp:
                if resp.status != 200:
                    return f"Failed to obtain weather data, status code:{resp.status}"
                weather_data = await resp.json()

        current_weather = weather_data.get("current_weather", {})
        temperature = current_weather.get("temperature")

        # Clothing Tips
        if temperature is None:
            outfit = "Unable to obtain temperature, unable to provide clothing suggestions"
        elif temperature < 5:
            outfit = "It is recommended to wear a down jacket or thick coat"
        elif 5 <= temperature < 15:
            outfit = "A coat or jacket is recommended."
        elif 15 <= temperature < 25:
            outfit = "Long sleeves or a light jacket are recommended."
        else:
            outfit = "The weather is hot, so it is recommended to wear short sleeves or cool clothes"

        return (
        f"📍 City: {city}\n"
        f"🌡 Current temperature: {temperature}°C\n"
        f"👕 Clothing suggestion: {outfit}\n"
        )

### 2. Creating a Custom Info Agent Using SpoonAI And Run Agent

##### Method 1: Inheriting from ToolCallAgent

SpoonAI provides a powerful base class called ToolCallAgent that enables your agent to automatically perform multi-step reasoning and call tools as needed. The agent will analyze the user's request, determine if a tool should be used, call it, and continue reasoning—until a final answer is returned.

###### Key Configuration Fields

You configure your Agent’s behavior using three key fields:

- name
  A unique identifier string for your agent.

- description
  A concise but descriptive explanation of what your agent does. This helps document the agent’s purpose and capabilities.

```python
description: str = (
    "A smart assistant that can:\n"
    "1. Retrieve monthly GitHub commit counts from a specific repository and branch.\n"
    "2. Provide current weather and outfit suggestions for a given city.\n"
)
````

- system_prompt
  This string sets the initial role and instructions for your Agent

```python
system_prompt: str = """
You are a helpful assistant with access to tools. You can:

1. Fetch the number of GitHub commits made this month.
2. Provide weather + outfit suggestions for a city.

Decide which tool to use, or reply directly if no tool is needed.

```

💡 It gives the Agent its identity, permissions, and behavior.

- next_step_prompt
  This is used between tool calls, guiding the Agent to decide what to do next.

```python
next_step_prompt: str = "Based on the previous result, decide what to do next."
```

💡 SpoonAI uses this prompt automatically after each tool.run() result.

-max_steps
How many times the Agent can loop through the reasoning → tool → reasoning cycle.
`max_steps: int = 5`

💡It helps avoid infinite loops, especially when tool output is ambiguous.

###### Registering Tools

You define tools using a ToolManager. telling the agent which functions it can call.

```python
avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
    GitHubCommitStatsTool(),
    SmartWeatherTool(),
]))
```

###### Here is a full example of a ToolCallAgent subclass implementation:

```python

from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from pydantic import Field

class MyInfoAgent(ToolCallAgent):
    """
    An intelligent assistant capable of performing useful information queries.
    Supports tools to retrieve GitHub statistics,
    and provide localized weather and air quality data with outfit suggestions.
    """

    name: str = "my_info_agent"
    description: str = (
        "A smart assistant that can:\n"
        "1. Retrieve monthly GitHub commit counts from a specific repository and branch.\n"
        "2. Provide current weather and outfit suggestions for a given city.\n"
    )

    system_prompt: str = """
    You are a helpful assistant with access to tools. You can:

    1. Fetch the number of commits made this month in a specific GitHub repository branch.
    2. Get current weather conditions and clothing suggestions for a specified city.

    For each user question, decide whether to invoke a tool or answer directly.
    If a tool's result isn't sufficient, analyze the result and guide the next steps clearly.
    """

    next_step_prompt: str = (
        "Based on the previous result, decide what to do next. "
        "If the result is incomplete, consider using another tool or asking for clarification."
    )

    max_steps: int = 5

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        GitHubCommitStatsTool(),
        SmartWeatherTool(),
    ]))
```

###### Running the Agent

Here’s a minimal example of how to run the agent:

`info_agent = MyInfoAgent(llm=ChatBot())`
💡We instantiate the custom agent class above
Optional parameters can be passed, instead of passing the default model_name: "clclde-3-7-sonnet-20250219", llm_provider: "anthropic"
model_name"gpt-4.5-preview" or model_name: "laude-3-7-sonnet-20250219"
llm_provider: "openapi" or llm_provider: "anthropic"

`question = "What is the weather like in Shanghai today?"`
Various questions to ask entered by the user

```python
from spoon_ai.chat import ChatBot
import asyncio

async def main():
    # Instantiate the agent with a language model
    info_agent = MyInfoAgent(llm=ChatBot())

    # Reset the agent’s memory (highly recommended before each run)
    info_agent.clear()

    # Provide a user question
    question = "What is the weather like in Shanghai today?"

    # Run the agent to get an intelligent response
    result = await info_agent.run(question)

    # Reset the agent’s memory (highly recommended before each run)
    info_agent.clear()

    # Provide a user question
    question = "How many commits are there in the master branch of neo-project/neo this month?"

    # Run the agent to get an intelligent response
    result = await info_agent.run(question)

    # Print the final response
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

```

##### Method 2：Create a custom Agent directly using ToolCallAgent

If you prefer not to define a new subclass for your Agent, SpoonAI also allows you to instantiate a ToolCallAgent directly and configure it inline. This method is quick and flexible, perfect for small-scale agents or rapid prototyping

Step 1 — Import and Prepare Tools
Make sure you've already implemented your tools and imported them properly:

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot
```

Step 2 — Instantiate the Agent
We build the agent inline by creating a ToolManager, setting the prompt, and instantiating the agent:

```python
async def create_custom_agent_directly():
    """Create a custom Agent directly using ToolCallAgent"""

    # Define available tools
    tool_manager = ToolManager([
        GitHubCommitStatsTool(),
        SmartWeatherTool()
    ])

    # Instantiate the Agent
    custom_agent = ToolCallAgent(
        name="direct_custom_agent",
        description="description: str = (
            "A smart assistant that can:\n"
            "1. Retrieve monthly GitHub commit counts from a specific repository and branch.\n"
            "2. Provide current weather and outfit suggestions for a given city.\n"",
        llm=ChatBot(),  # Use the default LLM
        avaliable_tools=tool_manager,
        system_prompt="""
            You are a helpful assistant with access to tools. You can:

            1. Fetch the number of commits made this month in a specific GitHub repository branch.
            2. Get current weather conditions and clothing suggestions for a specified city.

            For each user question, decide whether to invoke a tool or answer directly.
            If a tool's result isn't sufficient, analyze the result and guide the next steps clearly.
        """,
        max_steps=5
    )
    return custom_agent
```

⚙️ Parameters Explained Same as method 1
name: A short identifier string for the agent

description: A human-readable summary of what the agent can do.

llm: Specifies which model backend to use (defaults to SpoonAI’s ChatBot()).

avaliable_tools: A ToolManager object that defines all tools the agent can invoke.

system_prompt: Provides guidance and context to the language model about how to behave.

max_steps: Controls how many tool/reasoning loops the agent may perform before finalizing a response.

Step 3 — Run the Agent
To use this agent in practice, run it just like any other asynchronous function:

```python
import asyncio

async def main():
    custom_agent = await create_custom_agent_directly()
    response = await custom_agent.run("What is the weather like in Shanghai today? Should I wear a mask?")
    print(f"Answer: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())

```

### 2. Running our agent with a simple input

Let's try to run the agent with an input that requires a function call to give a suitable reply.
`custom_agent.run(What is the weather like in shanghai today?")`
When we run the code above, we see the response from OpenAI logged out to the console like this:

```python
=== Using InfoAssistantAgent ===
Answer: Step 1: Observed output of cmd smart_weather execution: 📍 City: Shanghai
🌡 Current temperature: 17.7°C
👕 Clothing suggestion: Long sleeves or a light jacket are recommended.

Step 2: The weather in Shanghai today is 17.7°C. It's recommended to wear long sleeves or a light jacket. Let me know if you need anything else!
Step 3: The previous result is complete and provides clear information about the current weather and clothing suggestions for Shanghai. No further action is needed unless you have additional questions or requests.
Step 4: The previous result is complete and provides all necessary information about the current weather and clothing suggestions for Shanghai. If you have any other questions or need further assistance, please let me know!
Step 5: The previous result is complete and clearly addresses the weather condition and clothing recommendation for Shanghai. No further action is required at this point. If you have any additional questions or need more information, please let me know!
Step 5: Stuck in loop. Resetting state.
```

`custom_agent.run("How many commits in neo-project/neo master branch this month?")`
When we run the code above, we see the response from OpenAI logged out to the console like this:

```python
Answer: Step 1: Observed output of cmd github_commits execution: Total commits in neo-project/neo (master) since 2025-04-01: 14
Step 2: The result is complete. There have been 14 commits in the master branch of the neo-project/neo repository since April 1, 2025. No further action is required.
Step 3: The previous result is complete and clearly answers your query. There have been 14 commits in the master branch of the neo-project/neo repository since April 1, 2025. No further action or clarification is necessary.
Step 4: The previous result is complete and fully addresses your query. There have been 14 commits in the master branch of the neo-project/neo repository since April 1, 2025. No additional steps or clarifications are necessary.
Step 5: The previous result is complete and clearly answers your question. There have been 14 commits in the master branch of the neo-project/neo repository since April 1, 2025. No further actions or clarifications are needed.
Step 5: Stuck in loop. Resetting state.
```

Complete code

```python
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.tools.base import BaseTool
from spoon_ai.chat import ChatBot
from datetime import datetime,timezone
from pydantic import Field

import aiohttp
import asyncio

# ---------------------------- 1. Smart Weather Tool ----------------------------
class SmartWeatherTool(BaseTool):
    """Smart Weather Tool"""
    name: str = "smart_weather"
    description: str = "Get weather and outfit suggestions for a city."
    parameters: dict = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g., 'Beijing'"
            }
        },
        "required": ["city"]
    }


    async def execute(self, city: str) -> str:
        # Step 1: Get latitude & longitude
        geocode_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"

        async with aiohttp.ClientSession() as session:
            async with session.get(geocode_url) as resp:
                if resp.status != 200:
                    return f"Failed to obtain the geographic location, status code:{resp.status}"
                geocode_data = await resp.json()

        if not geocode_data:
            return f"Unable to find geographic information for city {city}"

        lat = geocode_data[0]["lat"]
        lon = geocode_data[0]["lon"]

        # Step 2: Get weather info
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(weather_url) as resp:
                if resp.status != 200:
                    return f"Failed to obtain weather data, status code:{resp.status}"
                weather_data = await resp.json()

        current_weather = weather_data.get("current_weather", {})
        temperature = current_weather.get("temperature")

        # Clothing Tips
        if temperature is None:
            outfit = "Unable to obtain temperature, unable to provide clothing suggestions"
        elif temperature < 5:
            outfit = "It is recommended to wear a down jacket or thick coat"
        elif 5 <= temperature < 15:
            outfit = "A coat or jacket is recommended."
        elif 15 <= temperature < 25:
            outfit = "Long sleeves or a light jacket are recommended."
        else:
            outfit = "The weather is hot, so it is recommended to wear short sleeves or cool clothes"

        return (
        f"📍 City: {city}\n"
        f"🌡 Current temperature: {temperature}°C\n"
        f"👕 Clothing suggestion: {outfit}\n"
        )

# ---------------------------- 2. GitHub Commit Count Tool ----------------------------
class GitHubCommitStatsTool(BaseTool):
    name: str = "github_commits"
    description: str = "Get number of commits for a GitHub repo in a date range."
    parameters: dict = {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "e.g., 'neo-project/neo'"},
            "branch": {"type": "string", "description": "Branch name, e.g., 'master'"}
        },
        "required": ["repo", "branch"]
    }

    async def execute(self, repo: str, branch: str) -> str:
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(day=1).isoformat()
        until = now.isoformat()

        commits = []
        page = 1

        async with aiohttp.ClientSession() as session:
            while True:
                url = f"https://api.github.com/repos/{repo}/commits?sha={branch}&since={start_of_month}&until={until}&per_page=100&page={page}"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return f"Failed to fetch commits: {resp.status}"
                    data = await resp.json()
                    if not data:
                        break
                    commits.extend(data)
                    page += 1

        return f"Total commits in {repo} ({branch}) since {start_of_month[:10]}: {len(commits)}"

# ---------------------------- 3. Agent Definition ----------------------------
class MyInfoAgent(ToolCallAgent):
    """
    An intelligent assistant capable of performing useful information queries.
    Supports tools to retrieve GitHub statistics,
    and provide localized weather with outfit suggestions.
    """

    name: str = "my_info_agent"
    description: str = (
        "A smart assistant that can:\n"
        "1. Retrieve monthly GitHub commit counts from a specific repository and branch.\n"
        "2. Provide current weather and outfit suggestions for a given city.\n"
    )

    system_prompt: str = """
    You are a helpful assistant with access to tools. You can:

    1. Fetch the number of commits made this month in a specific GitHub repository branch.
    2. Get current weather conditions and clothing suggestions for a specified city.

    For each user question, decide whether to invoke a tool or answer directly.
    If a tool's result isn't sufficient, analyze the result and guide the next steps clearly.
    """

    next_step_prompt: str = (
        "Based on the previous result, decide what to do next. "
        "If the result is incomplete, consider using another tool or asking for clarification."
    )

    max_steps: int = 5

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        GitHubCommitStatsTool(),
        SmartWeatherTool(),
    ]))



async def main():
    # Create an InfoAssistantAgent instance
    info_agent = MyInfoAgent(llm=ChatBot())

    # Run the Agent
    print("=== Using InfoAssistantAgent ===")

    # Reset the Agent state
    info_agent.clear()

    response = await info_agent.run("What is the weather like in hongkong today?")
    print(f"Answer: {response}\n")

    info_agent.clear()
    response = await info_agent.run("How many commits are there in the master branch of neo-project/neo this month?")
    print(f"Answer: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
```
