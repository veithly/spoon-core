#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Custom Tools Example

This example demonstrates how to create more practical custom tools
that interact with external APIs, file systems, and databases.
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiohttp
import pandas as pd
from pydantic import Field

from spoon_ai.agents import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager
from spoon_ai.tools.base import BaseTool, ToolFailure, ToolResult


# 1. File System Tool
class FileSystemTool(BaseTool):
    """Tool for interacting with the file system"""
    name: str = "file_system"
    description: str = "Perform operations on the file system, such as reading, writing, and listing files."
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform (read, write, list)",
                "enum": ["read", "write", "list"]
            },
            "path": {
                "type": "string",
                "description": "The file or directory path"
            },
            "content": {
                "type": "string",
                "description": "The content to write (only for write operation)"
            }
        },
        "required": ["operation", "path"]
    }

    async def execute(self, operation: str, path: str, content: Optional[str] = None) -> str:
        """Execute file system operations"""
        try:
            path_obj = Path(path)
            
            if operation == "read":
                if not path_obj.exists():
                    return f"Error: File {path} does not exist"
                if not path_obj.is_file():
                    return f"Error: {path} is not a file"
                
                with open(path, 'r') as file:
                    file_content = file.read()
                return f"Content of {path}:\n\n{file_content}"
                
            elif operation == "write":
                if content is None:
                    return "Error: Content parameter is required for write operation"
                
                # Create parent directories if they don't exist
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w') as file:
                    file.write(content)
                return f"Successfully wrote {len(content)} characters to {path}"
                
            elif operation == "list":
                if not path_obj.exists():
                    return f"Error: Path {path} does not exist"
                if not path_obj.is_dir():
                    return f"Error: {path} is not a directory"
                
                files = [f.name for f in path_obj.iterdir()]
                return f"Contents of directory {path}:\n- " + "\n- ".join(files)
                
            else:
                return f"Error: Unsupported operation '{operation}'"
                
        except Exception as e:
            return f"Error performing {operation} on {path}: {str(e)}"


# 2. API Request Tool
class APIRequestTool(BaseTool):
    """Tool for making HTTP requests to external APIs"""
    name: str = "api_request"
    description: str = "Make HTTP requests to external APIs to fetch or send data."
    parameters: dict = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method (GET, POST, PUT, DELETE)",
                "enum": ["GET", "POST", "PUT", "DELETE"]
            },
            "url": {
                "type": "string",
                "description": "The URL to make the request to"
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers to include in the request"
            },
            "params": {
                "type": "object",
                "description": "Query parameters for the request"
            },
            "data": {
                "type": "object",
                "description": "Data to send in the request body (for POST/PUT)"
            }
        },
        "required": ["method", "url"]
    }

    async def execute(
        self, 
        method: str, 
        url: str, 
        headers: Optional[Dict] = None, 
        params: Optional[Dict] = None, 
        data: Optional[Dict] = None
    ) -> str:
        """Execute HTTP request"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = headers or {}
                params = params or {}
                
                if method == "GET":
                    async with session.get(url, headers=headers, params=params) as response:
                        response_text = await response.text()
                        return self._format_response(response, response_text)
                        
                elif method == "POST":
                    async with session.post(url, headers=headers, params=params, json=data) as response:
                        response_text = await response.text()
                        return self._format_response(response, response_text)
                        
                elif method == "PUT":
                    async with session.put(url, headers=headers, params=params, json=data) as response:
                        response_text = await response.text()
                        return self._format_response(response, response_text)
                        
                elif method == "DELETE":
                    async with session.delete(url, headers=headers, params=params) as response:
                        response_text = await response.text()
                        return self._format_response(response, response_text)
                        
                else:
                    return f"Error: Unsupported HTTP method '{method}'"
                    
        except Exception as e:
            return f"Error making {method} request to {url}: {str(e)}"
    
    def _format_response(self, response, text):
        """Format the HTTP response"""
        status = response.status
        try:
            # Try to parse as JSON for prettier output
            json_data = json.loads(text)
            formatted_data = json.dumps(json_data, indent=2)
            return f"Status: {status}\nResponse:\n{formatted_data}"
        except:
            # If not JSON, return as is
            return f"Status: {status}\nResponse:\n{text}"


# 3. Database Tool
class DatabaseTool(BaseTool):
    """Tool for interacting with SQLite databases"""
    name: str = "database"
    description: str = "Execute SQL queries on a SQLite database."
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform (query, execute)",
                "enum": ["query", "execute"]
            },
            "database_path": {
                "type": "string",
                "description": "Path to the SQLite database file"
            },
            "sql": {
                "type": "string",
                "description": "SQL statement to execute"
            }
        },
        "required": ["operation", "database_path", "sql"]
    }

    async def execute(self, operation: str, database_path: str, sql: str) -> str:
        """Execute database operations"""
        try:
            # Ensure the database directory exists
            db_path = Path(database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to the database
            conn = sqlite3.connect(database_path)
            
            if operation == "query":
                # For SELECT queries, return the results as a formatted table
                df = pd.read_sql_query(sql, conn)
                if df.empty:
                    result = "Query returned no results."
                else:
                    result = f"Query results:\n{df.to_string(index=False)}"
                
            elif operation == "execute":
                # For other SQL statements (INSERT, UPDATE, DELETE, CREATE, etc.)
                cursor = conn.cursor()
                cursor.execute(sql)
                conn.commit()
                affected_rows = cursor.rowcount
                result = f"SQL executed successfully. Affected rows: {affected_rows}"
                
            else:
                result = f"Error: Unsupported operation '{operation}'"
                
            conn.close()
            return result
            
        except Exception as e:
            return f"Error executing {operation} on {database_path}: {str(e)}"


# 4. Data Analysis Tool
class DataAnalysisTool(BaseTool):
    """Tool for analyzing data files"""
    name: str = "data_analysis"
    description: str = "Analyze data files (CSV, Excel) and perform basic statistical operations."
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the data file (CSV, Excel)"
            },
            "operation": {
                "type": "string",
                "description": "Analysis operation to perform",
                "enum": ["summary", "head", "tail", "describe", "columns", "count"]
            },
            "column": {
                "type": "string",
                "description": "Column name for column-specific operations"
            }
        },
        "required": ["file_path", "operation"]
    }

    async def execute(
        self, 
        file_path: str, 
        operation: str, 
        column: Optional[str] = None
    ) -> str:
        """Execute data analysis operations"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File {file_path} does not exist"
                
            # Load the data file
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return f"Error: Unsupported file format '{file_ext}'"
                
            # Perform the requested operation
            if operation == "summary":
                return f"File: {file_path}\nRows: {len(df)}\nColumns: {len(df.columns)}\nColumn names: {', '.join(df.columns)}"
                
            elif operation == "head":
                return f"First 5 rows of {file_path}:\n{df.head().to_string()}"
                
            elif operation == "tail":
                return f"Last 5 rows of {file_path}:\n{df.tail().to_string()}"
                
            elif operation == "describe":
                if column and column in df.columns:
                    return f"Statistics for column '{column}':\n{df[column].describe().to_string()}"
                else:
                    return f"Statistical summary of {file_path}:\n{df.describe().to_string()}"
                    
            elif operation == "columns":
                return f"Columns in {file_path}:\n- " + "\n- ".join(df.columns)
                
            elif operation == "count":
                if column and column in df.columns:
                    value_counts = df[column].value_counts()
                    return f"Value counts for column '{column}':\n{value_counts.to_string()}"
                else:
                    return f"Error: Column '{column}' not specified or not found"
                    
            else:
                return f"Error: Unsupported operation '{operation}'"
                
        except Exception as e:
            return f"Error analyzing {file_path}: {str(e)}"


# Create a custom Agent with the advanced tools
class DataAnalystAgent(ToolCallAgent):
    """Data Analyst Agent with advanced tools"""
    name: str = "data_analyst"
    description: str = "An agent that can analyze data, interact with databases, make API calls, and manage files"
    
    system_prompt: str = """You are a data analyst assistant that can help users with various data-related tasks.
    You can:
    1. Analyze data files (CSV, Excel)
    2. Execute SQL queries on databases
    3. Make API requests to fetch data
    4. Read and write files
    
    Use the appropriate tool based on the user's request. Be thorough in your analysis and explanations.
    """
    
    max_steps: int = 8
    
    # Define available tools
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        FileSystemTool(),
        APIRequestTool(),
        DatabaseTool(),
        DataAnalysisTool()
    ]))


# Run the example
async def main():
    # Create a sample CSV file for demonstration
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
    
    os.makedirs("examples/data", exist_ok=True)
    with open("examples/data/employees.csv", "w") as f:
        f.write(sample_data)
    
    # Create a sample database
    conn = sqlite3.connect("examples/data/sample.db")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        city TEXT,
        salary INTEGER
    )
    """)
    
    # Insert sample data
    conn.execute("DELETE FROM employees")  # Clear existing data
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
    
    print("Sample data files created.")
    
    # Create the agent
    agent = DataAnalystAgent(llm=ChatBot())
    
    # Run the agent with different queries
    queries = [
        "Analyze the employees.csv file and give me a summary",
        "What are the average salaries in the employees database?",
        "Read the first 5 rows of the employees.csv file",
        "How many employees are in each city according to the CSV file?"
    ]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        response = await agent.run(query)
        print(f"Response:\n{response}")
        
        # Reset agent state for the next query
        agent.clear()


if __name__ == "__main__":
    asyncio.run(main()) 