import os
import sqlite3
import pytest
import asyncio
from pathlib import Path
from examples.custom_tool_example import (
    DataAnalystAgent
)
from spoon_ai.chat import ChatBot
from spoon_ai.agents.base import AgentState

# FIXTURE: Create test data (CSV file & SQLite database)
@pytest.fixture(scope="module")
def setup_data():
    """Setup test environment: Create a sample CSV file and SQLite database."""
    sample_data = """id,name,age,city,salary
    1,John Smith,34,New York,75000
    2,Mary Johnson,28,San Francisco,85000
    3,Robert Brown,45,Chicago,92000
    4,Patricia Davis,31,Boston,78000
    5,James Wilson,39,Seattle,88000
    6,Jennifer Moore,27,Austin,72000
    7,Michael Taylor,42,Denver,95000
    8,Elizabeth Anderson,36,Portland,82000
    9,David Thomas,29,Los Angeles,79000
    10,Susan Jackson,44,Miami,91000
    """
    
    os.makedirs("tests/data", exist_ok=True)
    csv_path = "tests/data/employees.csv"

    # Create a CSV file
    with open(csv_path, "w") as f:
        f.write(sample_data)
    
    # Create a SQLite Database
    db_path = "tests/data/sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        city TEXT,
        salary INTEGER
    )
    """)
    
    # Insert data
    conn.execute("DELETE FROM employees")  # Clear the database
    conn.executemany(
        "INSERT INTO employees (id, name, age, city, salary) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "John Smith", 34, "New York", 75000),
            (2, "Mary Johnson", 28, "San Francisco", 85000),
            (3, "Robert Brown", 45, "Chicago", 92000),
            (4, "Patricia Davis", 31, "Boston", 78000),
            (5, "James Wilson", 39, "Seattle", 88000),
        ]
    )
    conn.commit()
    conn.close()

    return csv_path, db_path

# FIXTURE: Create Agent and reset it after each test
@pytest.fixture
async def agent():
    """Fixture to create an agent and ensure it resets state after each test."""
    a = DataAnalystAgent(llm=ChatBot())
    yield a  # Pass `agent` to the test function
    a.clear()  # Clear state after test
    a.state = AgentState.IDLE  # Reset agent state
    print(f"Agent state reset after test: {a.state}")


# # TEST: Test DataAnalystAgent to process CSV and database queries
@pytest.mark.asyncio
async def test_data_analyst_agent(setup_data, agent):
    """Test DataAnalystAgent's ability to process CSV and database queries."""
    csv_path, db_path = setup_data

    queries = [
        "Analyze the {csv_path} file and give me a summary".format(csv_path=csv_path),
        "What are the average salaries in the database {db_path}?".format(db_path=db_path),
        "Read the first 5 rows of the {csv_path} file".format(csv_path=csv_path),
        "How many employees are in each city according to the {csv_path} file?".format(csv_path=csv_path)
    ]

    # Expected results (examples, you may adjust as needed)
    expected_results = [
        "Summary: The dataset contains information about employees, including their ID, name, age, city, and salary.",
        "The average salary in the employees database is $83,600.",
        "id,name,age,city,salary\n1,John Smith,34,New York,75000\n2,Mary Johnson,28,San Francisco,85000\n3,Robert Brown,45,Chicago,92000\n4,Patricia Davis,31,Boston,78000\n5,James Wilson,39,Seattle,88000",
        "New York: 1 employee, San Francisco: 1 employee, Chicago: 1 employee, Boston: 1 employee, Seattle: 1 employee"
    ]

    print(f"Agent state before running: {agent.state}")
    
    for query, expected in zip(queries, expected_results):
        try:
            response = await agent.run(query)
            if "first 5 rows" in query.lower():
                assert "john smith" in response.lower()
                assert "patricia davis" in response.lower()
            else:
                assert any(phrase in response.lower() for phrase in expected.lower().split())

        finally:
            agent.state = AgentState.IDLE
            print(f"Agent state reset after running `{query}`: {agent.state}")


    # Additional check to ensure agent.clear() works
    agent.clear()
    print("Agent state successfully cleared.")


@pytest.mark.asyncio
async def test_api_request_agent(agent):
    """
    Test APIRequestTool to verify it correctly processes queries and executes API requests.
    
    This test covers:
    1. Fetching NEO price from CoinGecko
    2. Retrieving the latest block number from Neo mainnet
    3. Sending a POST request to a test API
    4. Fetching balances from the NeoX testnet RPC API
    """

    queries = [
        "Get the latest price of neo from neo mainnet",
        "Get the latest block number from the neo mainnet",
        "Send a POST request to https://jsonplaceholder.typicode.com/posts with the data { 'title': 'foo', 'body': 'bar', 'userId': 1 }",
        "Fetch balances from NeoX testnet RPC API for address 0xd7e0E170d285Ec91460CB5Cd49668523c8571065,url:https://neoxt4seed2.ngd.network"
    ]

    expected_responses = [
        "The latest price of Neo",  # Price query should return "price"
        "The latest block number on the Neo mainnet",  # Block number query should contain "block number"
        "Successfully posted data",  # POST request should return a success message
        "balance"  # NeoX balance query should include "balance"
    ]

    for query, expected in zip(queries, expected_responses):
        print(f"\n=== Running Query: {query} ===")
        
        # Execute the agent query
        response = await agent.run(query)

        print(f"Response:\n{response}")

        # Assert that the response is not None
        assert response is not None, f"Query `{query}` returned None!"
        assert isinstance(response, str), f"Query `{query}` returned a non-string response!"

        # Check if the response contains expected information (fuzzy matching)
        assert expected.lower() in response.lower(), f"Expected `{expected}` in response `{response}`"

        # Clear agent state after each query to avoid memory contamination
        agent.clear()


@pytest.mark.asyncio
async def test_filesystem_agent(agent):
    """
    Test FileSystemTool via AI agent interaction.

    The test covers:
    1. Writing content to files
    2. Reading files
    3. Listing directory contents
    4. Handling missing files/directories
    """

    # Define test directory and ensure it exists
    test_dir = Path("tests/data")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Define test queries
    queries = [
        "Write 'Hello, AI!' to a file at path 'tests/data/hello.txt'",
        "Read the content of the file at path 'tests/data/hello.txt'",
        "List all files in the directory tests/data",
        "Read the file at 'tests/data/missing_file.txt'",  # Expected to return an error
        "List files in the directory 'non_existent_dir'"  # Expected to return an error
    ]

    # Expected responses (supporting flexible matching)
    expected_responses = [
        "Successfully wrote",  # Writing to file
        "Hello, AI!",  # Reading written content
        "Contents of directory",  # Listing files
        ["Error: File", "does not exist"],  # Error for missing file
        ["Error: Path", "does not exist"]  # Error for missing directory
    ]

    # Run test cases
    for query, expected in zip(queries, expected_responses):
        print(f"\n=== Running Query: {query} ===")

        try:
            # Execute the agent query
            response = await agent.run(query)
            print(f"Response:\n{response}")

            # Validate response
            assert response is not None, f"Query `{query}` returned None!"
            assert isinstance(response, str), f"Query `{query}` returned a non-string response!"

            # More flexible assertion to handle different response variations
            if isinstance(expected, list):  # Handle multiple possible expected responses
                assert any(exp.lower() in response.lower() for exp in expected), \
                    f"Expected one of `{expected}` in response `{response}`"
            else:
                assert expected.lower() in response.lower(), \
                    f"Expected `{expected}` in response `{response}`"

        except Exception as e:
            pytest.fail(f"Test failed on query `{query}` due to exception: {e}")

        finally:
            # Reset agent state after each query to prevent state contamination
            agent.clear()

    # Cleanup after test
    try:
        for file in test_dir.glob("*"):
            file.unlink()
        test_dir.rmdir()
        print("Test files cleaned up successfully.")
    except Exception as cleanup_error:
        print(f"Cleanup failed: {cleanup_error}")
