import asyncio
import copy
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._tools_dict = {tool.name: tool for tool in self.tools}
        self._tool_schemas = [tool.schema for tool in self.tools] if self.tools else []
        self.state: dict[str, Any] = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request,
                             response: Response) -> Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )
        prepared_messages = self._prepare_messages(request.messages)
        tool_schemas = self._tool_schemas or None
        chunks = await client.chat.completions.create(
            deployment_name=deployment_name,
            messages=prepared_messages,
            tools=tool_schemas,
            stream=True,
        )

        tool_call_index_map: dict[int, dict[str, Any]] = {}
        content_parts: list[str] = []
        collected_attachments: list[dict[str, Any]] = []

        async for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            if delta.content:
                choice.append_content(delta.content)
                content_parts.append(delta.content)

            if delta.custom_content and delta.custom_content.attachments:
                for attachment in delta.custom_content.attachments:
                    attachment_dict = attachment.dict(exclude_none=True)
                    collected_attachments.append(attachment_dict)
                    attachment_kwargs: dict[str, Any] = {
                        "type": attachment_dict.get("type"),
                        "title": attachment_dict.get("title"),
                    }
                    if attachment_dict.get("reference_url"):
                        attachment_kwargs["reference_url"] = attachment_dict["reference_url"]
                    if attachment_dict.get("reference_type"):
                        attachment_kwargs["reference_type"] = attachment_dict["reference_type"]
                    if attachment_dict.get("url"):
                        attachment_kwargs["url"] = attachment_dict["url"]
                    elif attachment_dict.get("data"):
                        attachment_kwargs["data"] = attachment_dict["data"]
                    choice.add_attachment(**attachment_kwargs)

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index
                    if index is None:
                        continue
                    if tool_call_delta.id:
                        tool_call_index_map[index] = {
                            "index": index,
                            "id": tool_call_delta.id,
                            "type": tool_call_delta.type or "function",
                            "function": {
                                "name": tool_call_delta.function.name if tool_call_delta.function and tool_call_delta.function.name else "",
                                "arguments": tool_call_delta.function.arguments if tool_call_delta.function and tool_call_delta.function.arguments else "",
                            },
                        }
                    else:
                        tool_call = tool_call_index_map.get(index)
                        if not tool_call:
                            continue
                        function_delta = tool_call_delta.function
                        if function_delta:
                            if function_delta.name:
                                tool_call["function"]["name"] = function_delta.name
                            if function_delta.arguments:
                                existing_args = tool_call["function"].get("arguments", "")
                                tool_call["function"]["arguments"] = existing_args + function_delta.arguments

        content = "".join(content_parts)
        custom_content = None
        if collected_attachments:
            custom_content = CustomContent(attachments=collected_attachments)

        assistant_tool_calls = [
            ToolCall.validate(call_data)
            for _, call_data in sorted(tool_call_index_map.items(), key=lambda item: item[0])
        ]
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content or None,
            tool_calls=assistant_tool_calls or None,
            custom_content=custom_content,
        )

        if assistant_message.tool_calls:
            conversation_id = request.headers.get("x-conversation-id")
            if not conversation_id:
                raise ValueError("x-conversation-id header is required for tool calls")

            tasks = [
                self._process_tool_call(tool_call, choice, request.api_key, request.api_version, conversation_id)
                for tool_call in assistant_message.tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)
            history_entry = assistant_message.dict(exclude_none=True)
            self.state[TOOL_CALL_HISTORY_KEY].append(history_entry)
            self.state[TOOL_CALL_HISTORY_KEY].extend(
                [message for message in tool_messages if message]
            )
            return await self.handle_request(deployment_name, choice, request, response)

        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        state_history = copy.deepcopy(self.state.get(TOOL_CALL_HISTORY_KEY, []))
        unpacked_messages = unpack_messages(messages, state_history)
        prepared_messages: list[dict[str, Any]] = [
            {
                "role": Role.SYSTEM.value,
                "content": self.system_prompt,
            }
        ]
        prepared_messages.extend(unpacked_messages)

        for message in prepared_messages:
            print(json.dumps(message, ensure_ascii=False, default=str))

        return prepared_messages

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, api_version: str,
                                 conversation_id: str) -> dict[
        str, Any]:
        tool_name = tool_call.function.name
        stage = StageProcessor.open_stage(choice, name=tool_name)
        tool = self._tools_dict.get(tool_name)
        if not tool:
            StageProcessor.close_stage_safely(stage)
            raise ValueError(f"Tool '{tool_name}' is not registered")

        try:
            if tool.show_in_stage:
                stage.append_content("## Request arguments: \n")
                arguments = tool_call.function.arguments or "{}"
                try:
                    formatted_arguments = json.dumps(json.loads(arguments), indent=2)
                except json.JSONDecodeError:
                    formatted_arguments = arguments
                stage.append_content(f"```json\n\r{formatted_arguments}\n\r```\n\r")
                stage.append_content("## Response: \n")

            tool_message = await tool.execute(
                ToolCallParams(
                    tool_call=tool_call,
                    stage=stage,
                    choice=choice,
                    api_key=api_key,
                    api_version=api_version,
                    conversation_id=conversation_id,
                )
            )
        finally:
            StageProcessor.close_stage_safely(stage)

        return tool_message.dict(exclude_none=True)
