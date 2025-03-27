from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.tools.base import BaseTool

from pydantic import Field
import aiohttp
import asyncio
from spoon_ai.chat import ChatBot
from pydantic import Field
import datetime


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
        now = datetime.datetime.now(datetime.timezone.utc)
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

        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:00")
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


# ---------------------------- 4. Agent Definition ----------------------------
class MyInfoAgent(ToolCallAgent):
    """A custom agent that can get GitHub commits, fetch news, and convert currency. and Get weather and get air quality"""
    name: str = "my_info_agent"
    description: str = "A custom assistant for GitHub insights,currency conversion,and smart weather tool "

    system_prompt: str = """
    You are a helpful assistant that can:
    1. Fetch monthly commit count from a GitHub repo branch
    2. Convert an amount from one currency to another using live exchange rates.
    3. Get weather, outfit suggestions, and air quality analysis for a city.

    Choose the appropriate tool for each task and provide useful answers.
    If the user's request does not need a tool, reply directly.
    """

    next_step_prompt: str = "What should be the next step?"
    # Based on the previous tool output, what should we do now?
    max_steps: int = 5

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        GitHubCommitStatsTool(),
        ExchangeRateTool(),
        SmartWeatherTool()
    ]))


async def main():
    # Create an InfoAssistantAgent instance
    info_agent = MyInfoAgent(llm=ChatBot())

    # Run the Agent
    print("=== Using InfoAssistantAgent ===")

    # Reset the Agent state
    info_agent.clear()

    # Run the Agent again with a different question
    response = await info_agent.run("How many commits are there in the master branch of neo-project/neo this month?",)
    print(f"Answer: {response}\n")

    response = await info_agent.run("Convert 100 USD to RMB")
    print(f"Answer: {response}\n")

    response = await info_agent.run("what is the weather and pollution in Shanghai today? should i wear A mask?")
    print(f"Answer: {response}\n")


if __name__ == "__main__":
    asyncio.run(main()) 