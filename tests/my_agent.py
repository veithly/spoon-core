from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.tools.base import BaseTool

from spoon_ai.chat import ChatBot
from pydantic import Field

from dateutil.parser import parse
from gql import gql
from typing import Optional
from github_client import client
from datetime import datetime, timezone


import aiohttp
import asyncio


# ---------------------------- 1. GitHub Commit Count Tool ----------------------------
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


# ---------------------------- 2. Currency Exchange Tool ----------------------------
class ExchangeRateTool(BaseTool):
    """Currency Exchange Rate Tool"""
    name: str = "exchange_rate"
    description: str = "Convert an amount from one currency to another using live exchange rates."
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "from_currency": {
                "type": "string",
                "description": "The currency code to convert from (e.g., USD, EUR)"
            },
            "to_currency": {
                "type": "string",
                "description": "The currency code to convert to (e.g., CNY, JPY)"
            },
            "amount": {
                "type": "number",
                "description": "The amount to convert"
            }
        },
        "required": ["from_currency", "to_currency", "amount"]
    }

    async def execute(self, from_currency: str, to_currency: str, amount: float) -> str:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Failed to fetch exchange rate data for {from_currency}."
                data = await response.json()
                
                if data["result"] != "success":
                    return f"Exchange rate API returned an error: {data.get('error-type', 'Unknown error')}"
                
                rate = data["rates"].get(to_currency)
                if rate is None:
                    return f"Currency code {to_currency} not supported."
                
                converted = round(rate * amount, 2)
                return f"{amount} {from_currency} = {converted} {to_currency} (Rate: {rate})"


# ---------------------------- 3. Smart Weather Tool ----------------------------
class SmartWeatherTool(BaseTool):
    """Smart Weather Tool with outfit and pollution suggestions"""
    name: str = "smart_weather"
    description: str = "Get weather, outfit suggestions, and air quality analysis for a city."
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

        # Step 3: Get PM2.5 from air quality API
        air_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&hourly=pm2_5&timezone=auto"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(air_url) as resp:
                if resp.status != 200:
                    return f"Failed to obtain air quality data, status code: {resp.status}"
                air_data = await resp.json()

        now = datetime.now().strftime("%Y-%m-%dT%H:00")
        times = air_data.get("hourly", {}).get("time", [])
        pm_values = air_data.get("hourly", {}).get("pm2_5", [])
        pm25 = None
        if now in times:
            pm25 = pm_values[times.index(now)]

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

        # PM2.5 Recommendations
        if pm25 is None:
            aqi_tip = "PM2.5 data is not available, so it is recommended to refer to local monitoring stations or wear masks as a precaution."
        elif pm25 <= 12:
            aqi_tip = "The air quality is excellent, suitable for outdoor activities."
        elif pm25 <= 35.4:
            aqi_tip = "The air quality is good and outdoor activities are suitable."
        elif pm25 <= 55.4:
            aqi_tip = "Mild pollution, it is recommended to reduce outdoor activities and sensitive people should wear masks."
        elif pm25 <= 150.4:
            aqi_tip = "Moderate pollution, it is recommended to reduce going out and wear a mask."
        else:
            aqi_tip = "In case of severe pollution, try to avoid going out and wear a mask."

        return (
        f"📍 City: {city}\n"
        f"🌡 Current temperature: {temperature}°C\n"
        f"👕 Clothing suggestion: {outfit}\n"
        f"😷 PM2.5 current value: {pm25} µg/m³\n"
        f"🏃 Air quality suggestion: {aqi_tip}"
        )


# ---------------------------- 4. GitHub Issue Count Tool ----------------------------
class GitHubIssueStatsTool(BaseTool):
    name: str = "github_issue_stats"
    description: str = (
        "Query how many issues a GitHub user has opened or closed in a given repository "
        "between a custom date range (e.g. Jan 1 to Jan 31, 2024). "
        "Useful for analyzing contribution activity by user."
    )

    parameters: dict = {
        "type": "object",
        "properties": {
            "owner": {"type": "string", "description": "Repo owner, e.g. 'neo-project'"},
            "name": {"type": "string", "description": "Repo name, e.g. 'neo'"},
            "author": {"type": "string", "description": "GitHub username"},
            "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
            "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format (optional, defaults to today)"}
        },
        "required": ["owner", "name", "author", "start_date"]
    }

    async def execute(self, owner: str, name: str, author: str, start_date: str, end_date: Optional[str] = None) -> str:
        import pandas as pd

        start_dt = parse(start_date).astimezone(timezone.utc)
        end_dt = parse(end_date).astimezone(timezone.utc) if end_date else datetime.now(timezone.utc)

        if start_dt > datetime.now(timezone.utc):
            return f" Start date {start_date} is in the future. Please provide a past date."
        if end_dt > datetime.now(timezone.utc):
            end_dt = datetime.now(timezone.utc)

        query = gql('''
        query($owner: String!, $name: String!, $cursor: String, $start_time: DateTime) {
          repository(owner: $owner, name: $name) {
            issues(first: 100, after: $cursor, filterBy: {since: $start_time}) {
              edges {
                cursor
                node {
                  createdAt
                  closedAt
                  url
                  author { login }
                  state
                }
              }
            }
          }
        }
        ''')

        variables = {"owner": owner, "name": name, "cursor": None, "start_time": start_dt.isoformat()}
        opened_list = []
        closed_list = []

        while True:
            response = client.execute(query, variable_values=variables)
            edges = response["repository"]["issues"]["edges"]
            if not edges:
                break

            for edge in edges:
                node = edge["node"]
                if node["author"] and node["author"]["login"].lower() == author.lower():
                    created_at = parse(node["createdAt"]).astimezone(timezone.utc)
                    closed_at = parse(node["closedAt"]).astimezone(timezone.utc) if node.get("closedAt") else None

                    if start_dt <= created_at <= end_dt:
                        opened_list.append({"url": node["url"], "createdAt": created_at})
                    if closed_at and start_dt <= closed_at <= end_dt:
                        closed_list.append({"url": node["url"], "closedAt": closed_at})

            variables["cursor"] = edges[-1]["cursor"]

        if not opened_list and not closed_list:
            return f"No issues by {author} in {owner}/{name} from {start_date} to {end_dt.strftime('%Y-%m-%d')}."

        opened_count = len(opened_list)
        closed_count = len(closed_list)

        sample_url = (opened_list + closed_list)[0]['url']

        return (
            f"📊 GitHub Issue Stats for `{author}` in `{owner}/{name}` from {start_date} to {end_dt.strftime('%Y-%m-%d')}:\n"
            f"- 👤 {author}:\n"
            f"  - 🟢 Opened: {opened_count}\n"
            f"  - 🔴 Closed: {closed_count}\n"
            f"- 🔗 Sample issue: {sample_url}"
        )

    

# ---------------------------- 5. Agent Definition ----------------------------
class MyInfoAgent(ToolCallAgent):
    """
    An intelligent assistant capable of performing useful information queries.
    Supports tools to retrieve GitHub statistics, perform currency conversions,
    and provide localized weather and air quality data with outfit suggestions.
    """

    name: str = "my_info_agent"
    description: str = (
        "A smart assistant that can:\n"
        "1. Retrieve monthly GitHub commit counts from a specific repository and branch.\n"
        "2. Convert currency amounts between any two currencies using live exchange rates.\n"
        "3. Provide current weather, PM2.5 air quality, and outfit suggestions for a given city.\n"
        "4. Get statistics on issues created by a GitHub user in a specific repo during a given month or custom date range."
    )

    system_prompt: str = """
    You are a helpful assistant with access to tools. You can:
    
    1. Fetch the number of commits made this month in a specific GitHub repository branch.
    2. Convert amounts between currencies using up-to-date exchange rates.
    3. Get current weather conditions, PM2.5 air quality, and clothing suggestions for a specified city.
    4. Retrieve statistics about GitHub issues created by a particular user in a given repository during a certain time period.

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
        ExchangeRateTool(),
        SmartWeatherTool(),
        GitHubIssueStatsTool()
    ]))



async def main():
    # Create an InfoAssistantAgent instance
    info_agent = MyInfoAgent(llm=ChatBot())

    # Run the Agent
    print("=== Using InfoAssistantAgent ===")

    # Reset the Agent state
    info_agent.clear()

    # # Run the Agent again with a different question
    response = await info_agent.run("How many commits are there in the master branch of neo-project/neo this month?",)
    print(f"Answer: {response}\n")

    # 必须清理状态
    info_agent.clear()

    # response = await info_agent.run("Convert 100 USD to RMB")
    # print(f"Answer: {response}\n")

    response = await info_agent.run("what is the weather and pollution in Shanghai today? should i wear A mask?")
    print(f"Answer: {response}\n")


    # response = await info_agent.run("Query: How many issues did shargon create in neo-project/neo from 2025-02-01 to 2025-02-27?")
    # print(f"Answer: {response}\n")


if __name__ == "__main__":
    asyncio.run(main()) 