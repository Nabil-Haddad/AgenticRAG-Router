import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from ..config import Config

SERVER_MODULE = "src.retrieval_arena.mcp_server.server"


@dataclass
class RetrievalChoiceResult:
    question: str
    chosen_method: str | None
    tool_input: dict | None
    retrieved_results: list[dict] = field(default_factory=list)
    claude_final_text: str = ""


async def run_conversation( session: ClientSession, anthropic_client: anthropic.Anthropic, question: str, model: str, top_k: int = 5, ) -> RetrievalChoiceResult:
    mcp_tools = (await session.list_tools()).tools
    tools = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in mcp_tools]

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "any", "disable_parallel_tool_use": True},
        messages=[{
            "role": "user",
            "content": f"Answer using exactly one retrieval tool with top_k={top_k}: {question}",
        }],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        text = next((b.text for b in response.content if b.type == "text"), "")
        return RetrievalChoiceResult(question=question, chosen_method=None, tool_input=None, claude_final_text=text)

    tool_result = await session.call_tool(tool_use.name, tool_use.input)
    retrieved = json.loads(tool_result.content[0].text)

    followup = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result.content[0].text}
            ]},
        ],
    )
    final_text = next((b.text for b in followup.content if b.type == "text"), "")

    return RetrievalChoiceResult(
        question=question,
        chosen_method=tool_use.name,
        tool_input=tool_use.input,
        retrieved_results=retrieved,
        claude_final_text=final_text,
    )


async def ask_with_mcp_tools( question: str, model: str = Config.ANTHROPIC_MODEL, top_k: int = 5, server_env: dict[str, str] | None = None, anthropic_client: anthropic.Anthropic | None = None, ) -> RetrievalChoiceResult:
    client = anthropic_client or anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    env = {**get_default_environment(), **(server_env or {})}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(Config.project_root),
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await run_conversation(session, client, question, model, top_k)

