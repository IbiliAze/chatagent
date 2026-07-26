import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

from core.config.settings import get_settings
from core.logging.logger import logger


class McpClient:
  """Loads the tools of one remote MCP server over streamable HTTP.

  langchain-mcp-adapters is async only, but the researcher graph is driven
  synchronously, so every await is run on a private event loop in a background
  thread. asyncio.run would not do: it raises when the caller is already inside
  a running loop, which it is whenever the graph is invoked from a FastAPI route.

  The server is stateless, so the adapters open and close a session per tool
  call and there is no connection to keep alive between them.
  """

  def __init__(
    self,
    name: str,
    url: str | None = None,
    token: str | None = None,
    timeout: float | None = None,
  ) -> None:
    settings = get_settings()
    self.name = name
    self.timeout = settings.mcp_timeout_seconds if timeout is None else timeout
    token = settings.mcp_token if token is None else token

    connection: StreamableHttpConnection = {
      'transport': 'streamable_http',
      'url': url or settings.mcp_url,
      'headers': {'Authorization': f'Bearer {token}'} if token else None,
      'timeout': self.timeout,
    }
    self.client = MultiServerMCPClient({name: connection})

    self._loop: asyncio.AbstractEventLoop | None = None
    self._loop_lock = threading.Lock()

  def load_tools(self) -> list[BaseTool]:
    """Load every tool the server publishes.

    Makes a tools/list call, so load once at startup rather than per request.
    """
    return [self._as_sync_tool(t) for t in self._list_tools()]

  def load_tool(self, remote_name: str, name: str | None = None) -> BaseTool:
    """Load a single tool by its name on the server, optionally renaming it."""
    remote_tools = self._list_tools()

    for remote_tool in remote_tools:
      if remote_tool.name == remote_name:
        return self._as_sync_tool(remote_tool, name)

    published = ', '.join(sorted(t.name for t in remote_tools)) or 'none'
    raise ValueError(
      f'MCP server {self.name!r} publishes no tool {remote_name!r} (has: {published})'
    )

  def _list_tools(self) -> list[BaseTool]:
    """Read the server's tool list.

    A refused or unreachable server surfaces as an anyio ExceptionGroup from
    deep inside the transport, which says nothing about which server failed, so
    name it here: this runs at startup and the message is what gets read.
    """
    try:
      return self._run(self.client.get_tools(server_name=self.name))
    except Exception as e:
      url = self.client.connections[self.name].get('url', 'unknown url')
      raise RuntimeError(
        f'Could not read the tool list of MCP server {self.name!r} at {url}'
      ) from e

  def close(self) -> None:
    """Stop the background event loop. Safe to call without ever having run one."""
    with self._loop_lock:
      if self._loop is not None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None

  def _as_sync_tool(self, remote: BaseTool, name: str | None = None) -> StructuredTool:
    """Wrap an async adapter tool so it can also be called synchronously.

    The args schema and description come from the server, so a tool's parameters
    stay in step with it without being restated here.
    """
    if not isinstance(remote, StructuredTool) or remote.coroutine is None:
      raise TypeError(f'MCP tool {remote.name!r} is not an async StructuredTool')

    coroutine = remote.coroutine
    tool_name = name or remote.name

    def unavailable() -> tuple[str, None]:
      """Report a broken server to the model instead of failing the graph.

      Only reached for transport and timeout failures: a tool that ran and
      returned isError already arrives as readable content.
      """
      logger.exception(f'MCP tool {remote.name!r} on {self.name!r} failed')
      return (f'{tool_name} is temporarily unavailable.', None)

    async def call(**kwargs: Any) -> Any:
      try:
        return await coroutine(**kwargs)
      except ToolException:
        raise
      except Exception:
        return unavailable()

    def call_sync(**kwargs: Any) -> Any:
      # Drives call() rather than the adapter's coroutine directly: an async def
      # is a Coroutine, which is what run_coroutine_threadsafe accepts, whereas
      # StructuredTool.coroutine is only typed (and only guaranteed) Awaitable.
      # The guard here is for _run itself, e.g. the backstop timeout firing.
      try:
        return self._run(call(**kwargs))
      except ToolException:
        raise
      except Exception:
        return unavailable()

    return StructuredTool(
      name=tool_name,
      description=remote.description,
      args_schema=remote.args_schema,
      func=call_sync,
      coroutine=call,
      # The adapters' coroutine resolves to (content, artifact), which is what
      # unavailable() imitates on failure.
      response_format='content_and_artifact',
      metadata=remote.metadata,
      handle_tool_error=remote.handle_tool_error,
    )

  def _run[T](self, coro: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine to completion on the background loop.

    The transport enforces its own HTTP timeout; the margin here is only a
    backstop so a wedged session cannot block the graph forever.
    """
    future = asyncio.run_coroutine_threadsafe(coro, self._get_loop())
    return future.result(timeout=self.timeout + 5)

  def _get_loop(self) -> asyncio.AbstractEventLoop:
    """Return the background loop, starting its thread on first use."""
    with self._loop_lock:
      if self._loop is None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(
          target=self._loop.run_forever, name=f'mcp-{self.name}', daemon=True
        ).start()
      return self._loop
