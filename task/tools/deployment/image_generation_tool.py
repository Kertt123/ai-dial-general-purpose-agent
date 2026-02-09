from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        msg = await super()._execute(tool_call_params)

        if msg.custom_content and msg.custom_content.attachments:
            for attachment in msg.custom_content.attachments:
                if attachment.type in ("image/png", "image/jpeg"):
                    tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")

            if not msg.content:
                msg.content = StrictStr(
                    'The image has been successfully generated according to request and shown to user!')

        return msg

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        return "Generates an image based on a textual description. Use this tool when the user asks to create, draw, or generate an image."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated.",
                },
                "quality": {
                    "type": "string",
                    "enum": ["standard", "hd"],
                    "description": "The quality of the image that will be generated. 'hd' creates images with finer details and greater consistency across the image.",
                },
                "style": {
                    "type": "string",
                    "enum": ["natural", "vivid"],
                    "description": "The style of the generated images. 'vivid' causes the model to lean towards generating hyper-real and dramatic images. 'natural' causes the model to produce more natural, less hyper-real looking images.",
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1792x1024", "1024x1792"],
                    "description": "The size of the generated images.",
                },
            },
            "required": ["prompt"],
        }
