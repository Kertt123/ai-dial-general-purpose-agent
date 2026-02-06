from typing import Any, Optional

from pydantic import BaseModel, Field


class MCPToolModel(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
