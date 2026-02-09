import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are an AI assistant that answers questions based on the provided context.
The context below contains retrieved chunks of a document.
Your task is to synthesize this information to answer the user's question.
If the answer is not present in the context, state that you cannot answer the question based on the provided document.

Context:
---
{context}
---
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.model = SentenceTransformer(
            model_name_or_path='all-MiniLM-L6-v2'
            # device='cpu'
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "semantic_search_in_document"

    @property
    def description(self) -> str:
        return (
            "Performs a semantic search within a specified document to find answers to questions. "
            "Use this tool when you need to answer a question based on the content of a file. "
            "Provide the user's question or search query and the URL of the file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document"
                },
                "file_url": {
                    "type": "string",
                    "description": "The URL of the file to search within"
                }
            },
            "required": ["request", "file_url"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = json.loads(tool_call_params.tool_call.function.arguments)
        request = args["request"]
        file_url = args["file_url"]
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)

        if cached_data:
            index, chunks = cached_data
        else:
            extractor = DialFileContentExtractor(self.endpoint, tool_call_params.api_key)
            text_content = extractor.extract_text(file_url)

            if not text_content:
                stage.append_content("Could not extract content from the file.\n\r")
                return "Error: File content not found or could not be extracted."

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings, dtype='float32'))
            self.document_cache.set(cache_document_key, index, chunks)

        query_embedding = self.model.encode([request])
        distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=3)

        retrieved_chunks = [chunks[i] for i in indices[0]]

        augmented_prompt = self.__augmentation(request, retrieved_chunks)

        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        dial = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version=tool_call_params.api_version,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(context="\n---\n".join(retrieved_chunks))},
            {"role": "user", "content": request}
        ]

        full_response = ""
        async for chunk in await dial.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True,
        ):
            if chunk.choices and chunk.choices[0].delta.content is not None:
                stage.append_content(chunk.choices[0].delta.content)
                full_response += chunk.choices[0].delta.content

        return full_response

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        return f"Question: {request}\n\nContext:\n" + "\n---\n".join(chunks)
