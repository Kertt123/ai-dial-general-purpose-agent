import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionTool(BaseTool):
    """
    Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
    PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
    USAGE: Start with page=1 (by default)
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "get_file_content"

    @property
    def description(self) -> str:
        return (
            "Extracts text content from files. Supported formats: PDF (text only), TXT, CSV (as a markdown table), HTML/HTM."
            "For files larger than 10,000 characters, pagination is enabled. "
            "The response for a paginated file will include '**Page #X. Total pages: Y**' at the end. "
            "To navigate through pages, use the 'page' parameter, starting with page=1 by default."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "The URL of the file to extract content from."
                },
                "page": {
                    "type": "integer",
                    "default": 1,
                    "description": "For large documents, pagination is enabled. Each page consists of 10000 characters."
                }
            },
            "required": ["file_url"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        file_url = args.get("file_url")
        page = args.get("page", 1)
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**File URL**: {file_url}\n\r")
        if page > 1:
            stage.append_content(f"**Page**: {page}\n\r")
        stage.append_content("## Response: \n")

        if not file_url:
            content = "Error: file_url is missing."
            stage.append_content(f"```text\n\r{content}\n\r```\n\r")
            return content

        extractor = DialFileContentExtractor(self.endpoint, tool_call_params.api_key)
        content = extractor.extract_text(file_url)

        if not content:
            content = "Error: File content not found."
            stage.append_content(f"```text\n\r{content}\n\r```\n\r")
            return content

        if len(content) > 10000:
            page_size = 10000
            total_pages = (len(content) + page_size - 1) // page_size
            if page < 1:
                page = 1

            if page > total_pages:
                content = f"Error: Page {page} does not exist. Total pages: {total_pages}"
            else:
                start_index = (page - 1) * page_size
                end_index = start_index + page_size
                page_content = content[start_index:end_index]
                content = f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"

        stage.append_content(f"```text\n\r{content}\n\r```\n\r")
        return content
