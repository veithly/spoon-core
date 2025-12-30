"""Example: Using Agent with three image upload methods

This example demonstrates how to use add_message_with_image() with:
1. Local image (base64 encoded)
2. Network image (external URL)
3. Data URL (base64 embedded in URL format)
"""

import asyncio
import base64
from pathlib import Path
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager

class VisionAgent(ToolCallAgent):
    """Agent that supports image analysis"""
    name: str = "vision_agent"
    system_prompt: str = """
    You are a vision analysis assistant. You can analyze images and answer questions about them.
    When a user provides an image, describe what you see and answer their questions.
    """
    max_steps: int = 5
    available_tools: ToolManager = ToolManager([])

async def main():
    """Demonstrate three image upload methods"""
    agent = VisionAgent(llm=ChatBot())
    
    print("=" * 60)
    print("Method 1: Local Image (Base64)")
    print("=" * 60)
    
    # Method 1: Local image file -> Base64 encoding
    image_path = Path("test_image.png")
    if image_path.exists():
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        await agent.add_message_with_image(
            "user",
            "What's in this local image? Describe the main objects and colors.",
            image_data=image_base64,  # Method 1: Base64
            image_media_type="image/png"
        )
        
        response1 = await agent.run()
        print(f"Response:\n{response1}\n")
    else:
        print(f"⚠️  Image file not found: {image_path}")
        print("   Please create test_image.png or update the path.\n")
    
    print("=" * 60)
    print("Method 2: Network Image (External URL)")
    print("=" * 60)
    
    # Method 2: External URL (network image)
    network_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4DAVNYMwi5Crnz0e4JHOu98crw6mz12wNujkIz_OVZ3AzRPNFX0UUqAI&s"
    
    await agent.add_message_with_image(
        "user",
        "What's in this network image? Describe the main objects and colors.",
        image_url=network_url  # Method 2: External URL
    )
    
    response2 = await agent.run()
    print(f"Response:\n{response2}\n")
    
    print("=" * 60)
    print("Method 3: Data URL (Base64 embedded in URL)")
    print("=" * 60)
    
    # Method 3: Data URL (base64 embedded in URL format)
    # First, read local image and encode to base64 (same as Method 1)
    if image_path.exists():
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Convert base64 to data URL format
        data_url = f"data:image/png;base64,{image_base64}"
        
        await agent.add_message_with_image(
            "user",
            "What's in this data URL image? Describe the main objects and colors.",
            image_url=data_url  # Method 3: Data URL
        )
        
        response3 = await agent.run()
        print(f"Response:\n{response3}\n")
    else:
        print(f"⚠️  Image file not found: {image_path}")
        print("   Cannot demonstrate Method 3 (requires Method 1's base64 data).\n")

if __name__ == "__main__":
    asyncio.run(main())

