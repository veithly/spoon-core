# SpoonAI Examples

This directory contains example code demonstrating how to use SpoonAI's agent framework.

## Prerequisites

Before running these examples, make sure you have:

1. Installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your API keys as environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   # or
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Available Examples

### 1. Basic Custom Agent Example

File: `custom_agent_example.py`

This example demonstrates how to create simple custom tools and agents. It includes:
- Creating basic tools (web search, calculator, weather info)
- Creating a custom agent by inheriting from ToolCallAgent
- Creating a custom agent directly using ToolCallAgent
- Running agents with different queries

To run this example:
```bash
python examples/custom_agent_example.py
```

### 2. Advanced Custom Tools Example

File: `custom_tool_example.py`

This example demonstrates more practical custom tools that interact with:
- File systems (reading, writing, listing files)
- External APIs (making HTTP requests)
- Databases (executing SQL queries)
- Data files (analyzing CSV/Excel files)

It also shows how to create a specialized data analyst agent that uses these tools.

To run this example:
```bash
python examples/custom_tool_example.py
```

## Creating Your Own Agents

These examples serve as templates for creating your own custom agents. To create your own agent:

1. Define your custom tools by inheriting from `BaseTool`
2. Create a tool manager with your tools
3. Create your agent by either:
   - Inheriting from `ToolCallAgent` (or another agent class)
   - Directly instantiating `ToolCallAgent` with your tools

For more detailed information, refer to the main [SpoonAI documentation](../README.md).

## Best Practices

When creating custom tools and agents:

1. Provide clear, detailed descriptions for your tools
2. Define parameter schemas carefully with proper types and descriptions
3. Handle errors gracefully in your tool execution
4. Set appropriate system prompts to guide the agent's behavior
5. Use a reasonable `max_steps` value to prevent infinite loops

## Troubleshooting

If you encounter issues:

1. Check that your API keys are set correctly
2. Ensure all dependencies are installed
3. Look for error messages in the tool execution output
4. Try simplifying your queries or tools to isolate the problem 