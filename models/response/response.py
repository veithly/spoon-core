import asyncio

from fastapi.responses import StreamingResponse


class OpenAIStreamResponse(StreamingResponse):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def listen_for_disconnect(self, receive):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
