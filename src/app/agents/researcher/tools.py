"""Tool definitions available to the researcher agent's specialists."""

from langchain_core.tools import BaseTool, StructuredTool

from app.mcp.mcp_client import McpClient
from app.rag.rag import Rag
from core.config.settings import get_settings
from core.logging.logger import logger


class ResearcherTools:
    """Assembles the RAG and MCP-backed tools the specialists can call."""

    def __init__(self, rag: Rag, mcp_client: McpClient) -> None:
        settings = get_settings()
        self.remote_name = settings.mcp_remote_name
        self.rag = rag
        self.mcp_client = mcp_client
        self.get_relevant_documents = StructuredTool.from_function(
            self._get_relevant_documents, name='get_relevant_documents'
        )

    def load_tools(self) -> list[BaseTool]:
        """Every tool the researcher can call.

        Reads the MCP server for the company knowledge schema, so call this once at
        startup rather than per request.
        """
        return [self.get_relevant_documents, self._search_company_knowledge()]

    def _get_relevant_documents(self, query: str) -> str:
        """Search for information relevant to the query from RAG."""
        logger.debug('RAG invokation triggered by query', extra={'query': query})
        return self.rag.ask(query)

    def _search_company_knowledge(self) -> BaseTool:
        """Load the tool that searches Eight Mile's company knowledge pages.

        The returned tool answers questions about:
        - services
        - technical capabilities
        - engagement models
        - contact information
        - representative authority
        - supported client enquiries

        Talks to the MCP server to read the tool's schema, so load it once at
        startup rather than per request.
        """
        logger.debug('Loading search_company_knowledge tool from MCP server')
        return self.mcp_client.load_tool(
            self.remote_name, name='search_company_knowledge'
        )
