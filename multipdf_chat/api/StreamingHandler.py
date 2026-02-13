import asyncio
from langchain.callbacks.base import AsyncCallbackHandler

class StreamingHandler(AsyncCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
    
    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put(token)

    async def on_llm_error(self, error: Exception, **kwargs):
        await self.queue.put(f"Error: {str(error)}")
        await self.queue.put(None) # close the stream 

    async def on_llm_end(self, *args, **kwargs):
        await self.queue.put(None)