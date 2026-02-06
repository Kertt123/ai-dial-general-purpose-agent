import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, MessageContentTextPart
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    @property
    def system_prompt(self) -> str | None:
        return None

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.pop("prompt")

        dial = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key
        )

        messages = []
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        messages.append(Message(role=Role.USER, content=prompt))

        content = ""
        attachments = []
        async for chunk in await dial.chat.completions.create(
            deployment=self.deployment_name,
            messages=messages,
            stream=True,
            extra_body={"custom_fields": arguments},
            **self.tool_parameters
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].delta.custom_content and chunk.choices[0].delta.custom_content.attachments:
                attachments.extend(chunk.choices[0].delta.custom_content.attachments)

        custom_content = CustomContent(attachments=attachments) if attachments else None

        return Message(
            role=Role.TOOL,
            content=[MessageContentTextPart(type='text', text=StrictStr(content))],
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id)
        )
