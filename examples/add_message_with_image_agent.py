"""Example: Using Agent with multiple file upload methods

This example demonstrates different ways to add files to agent messages:
1. Local image with manual base64 encoding (using add_message_with_image)
2. Network image (external URL using add_message_with_image)
3. Data URL (base64 embedded in URL format using add_message_with_image)
4. Local image file (using add_message_with_image_file - convenience method)
5. PDF file (using add_message_with_pdf_file - convenience method)
6. Universal file method (using add_message_with_file - auto-detects file type)
"""

import asyncio
import base64
from pathlib import Path
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager

class FileAnalysisAgent(ToolCallAgent):
    """Agent that supports file analysis"""
    name: str = "file_analysis_agent"
    system_prompt: str = """
    You are a file analysis assistant. You can analyze images, PDFs, and documents.
    When a user provides a file, analyze it and answer their questions.
    """
    max_steps: int = 5
    available_tools: ToolManager = ToolManager([])

async def main():
    """Demonstrate multiple file upload methods"""
    agent = FileAnalysisAgent(llm=ChatBot())
    
    # Find local image file
    image_paths = Path("./picture/image.png")
    if not image_paths.exists():
        raise FileNotFoundError(f"Image file not found: {image_paths}")
    image_path = image_paths
    
    # Find README file
    readme_path = Path("./README.md")
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")

    print("=" * 60)
    print("Method 1: Local Image (Manual Base64 Encoding)")
    print("=" * 60)
    print("Using: add_message_with_image() with image_data parameter\n")
    
    # # Method 1: Manual base64 encoding (for more control)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    await agent.add_message_with_image(
        "user",
        "What's in this manually encoded image? Describe the main objects and colors.",
        image_data=image_base64,  # Method 1: Manual base64
        image_media_type="image/png"
    )
    
    response1 = await agent.run()
    print(f"Response:\n{response1}\n")

    
    print("=" * 60)
    print("Method 2: Network Image (External URL)")
    print("=" * 60)
    print("Using: add_message_with_image() with image_url parameter\n")
    
    # # Method 2: External URL (network image)
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
    print("Using: add_message_with_image() with image_url parameter (data URL)\n")
    
    # # Method 3: Data URL (base64 embedded in URL format)
    if image_path and image_path.exists():
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Convert base64 to data URL format
        mime_type = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        data_url = f"data:image/png;base64,{image_base64}"
        
        await agent.add_message_with_image(
            "user",
            "What's in this data URL image? Describe the main objects and colors.",
            image_url=data_url  # Method 3: Data URL
        )
        
        response3 = await agent.run()
        print(f"Response:\n{response3}\n")
    else:
        print(f"⚠️  Image file not found")
        print("   Cannot demonstrate Method 3 (requires local image file).\n")
    
    print("=" * 60)
    print("Method 4: Local Image File (Convenience Method)")
    print("=" * 60)
    print("Using: add_message_with_image_file()")
    print("Automatically handles base64 encoding and MIME type detection\n")
    
    # # Method 4: Using the convenience method for local image files
    if image_path and image_path.exists():
        await agent.add_message_with_image_file(
            "user",
            "What's in this local image file? Describe the main objects and colors.",
            file_path=str(image_path)  # Method 4: Convenience method
        )
        
        response4 = await agent.run()
        print(f"Response:\n{response4}\n")
    else:
        print(f"⚠️  Image file not found")
        print("   Tried paths:", [str(p) for p in image_paths])
        print()
    
    print("=" * 60)
    print("Method 5: PDF File (Convenience Method)")
    print("=" * 60)
    print("Using: add_message_with_pdf_file()")
    print("Automatically handles base64 encoding for PDF files")
    print("Note: This method is specifically for PDF files.\n")
    
    # Method 5: Using the convenience method for PDF files
    # Note: add_message_with_pdf_file requires PDF, so using actual PDF file
    await agent.add_message_with_pdf_file(
        "user",
        "Please summarize the key points in this PDF document.",
        file_path=str(readme_path)  # Method 5: PDF convenience method
    )
    
    response5 = await agent.run()
    print(f"Response:\n{response5}\n")

    
    print("=" * 60)
    print("Method 6: Universal File Method (Auto-detects file type)")
    print("=" * 60)
    print("Using: add_message_with_file() - works with images, PDFs, and text files")
    print("Demonstrating with README.md (auto-detects and routes to add_message_with_document)\n")
    
    # Method 6: Universal file method (auto-detects file type) - using README
    # This will automatically route to add_message_with_document for .md files

    await agent.add_message_with_file(
        "user",
        "Please summarize the main content of this README file.",
        file_path=str(readme_path)  # Method 6: Universal method with README
    )
    
    response6 = await agent.run()
    print(f"Response:\n{response6}\n")


if __name__ == "__main__":
    asyncio.run(main())

