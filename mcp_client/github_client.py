import asyncio
import threading
from typing import Any, Dict, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class GitHubMcpClient:
    """
    Safe synchronous wrapper for MCP using:
      - one dedicated event loop running in a background thread
      - one persistent stdio_client + ClientSession
    Avoids AnyIO cancel-scope errors caused by calling asyncio.run() repeatedly.
    """

    def __init__(self, server_params: StdioServerParameters):
        self._server_params = server_params

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._stdio_cm = None
        self._session: Optional[ClientSession] = None
        self._started = threading.Event()
        self._lock = threading.Lock()

        self._start_loop_thread()

    # ---------- public API ----------

    def list_tools(self):
        return self._submit(self._list_tools_async())

    def call_tool(self, name: str, arguments: Dict[str, Any]):
        return self._submit(self._call_tool_async(name, arguments))

    def close(self) -> None:
        with self._lock:
            if self._loop is None:
                return
            fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
        try:
            fut.result(timeout=10)
        finally:
            # stop loop thread
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None

    # ---------- internal ----------

    def _start_loop_thread(self) -> None:
        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.create_task(self._connect_async())
            self._started.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)

    def _submit(self, coro):
        if self._loop is None:
            raise RuntimeError("MCP client loop not running.")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=60)

    async def _connect_async(self) -> None:
        # open transport + session once
        self._stdio_cm = stdio_client(self._server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def _ensure_ready(self) -> None:
        # wait until connect has completed
        while self._session is None:
            await asyncio.sleep(0.01)

    async def _list_tools_async(self):
        await self._ensure_ready()
        return await self._session.list_tools()

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any]):
        await self._ensure_ready()
        return await self._session.call_tool(name, arguments)

    async def _close_async(self):
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None
