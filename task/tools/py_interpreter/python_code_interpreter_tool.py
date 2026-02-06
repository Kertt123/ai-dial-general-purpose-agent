import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from aidial_sdk.chat_completion import Message, Attachment

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self.dial_endpoint = dial_endpoint
        self.mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break

        if self._code_execute_tool is None:
            raise ValueError(f"Tool with name '{tool_name}' not found in MCP server.")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        mcp_client = await MCPClient.create(mcp_url)
        tools = await mcp_client.get_tools()
        return cls(mcp_client, tools, tool_name, dial_endpoint)

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        code = args["code"]
        session_id = args.get("session_id")
        stage = tool_call_params.stage

        await stage.append_content("## Request arguments: \n")
        await stage.append_content(f"```python\n{code}\n```\n")
        if session_id:
            await stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            await stage.append_content("New session will be created\n\r")

        response_str = await self.mcp_client.call_tool(self.name, args)
        response_data = json.loads(response_str)
        execution_result = _ExecutionResult.model_validate(response_data)

        if execution_result.files:
            dial = Dial(self.dial_endpoint, tool_call_params.api_key)
            files_home = await dial.get_files_home()
            for file in execution_result.files:
                file_name = file.name
                mime_type = file.mime_type
                resource_content = await self.mcp_client.get_resource(file.url)

                if mime_type.startswith("text/") or mime_type in ['application/json', 'application/xml']:
                    file_content = resource_content.encode('utf-8')
                else:
                    file_content = base64.b64decode(resource_content)

                upload_path = files_home / file_name
                await dial.upload(upload_path.as_posix(), file_content, mime_type)

                attachment = Attachment(type=mime_type, title=file_name, url=upload_path.as_posix())
                await stage.append_content(f"Generated file: {file_name}")
                tool_call_params.choice.add_attachment(attachment)

        if execution_result.output:
            for i, out in enumerate(execution_result.output):
                if len(out) > 1000:
                    execution_result.output[i] = out[:1000] + "..."

        result_json = execution_result.model_dump_json(indent=2)
        await stage.append_content(f"```json\n{result_json}\n```\n")

        return result_json
