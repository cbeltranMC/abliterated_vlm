"""Generate the OpenCode provider entry for the laptop.

The laptop reaches the server through an SSH tunnel, so the base URL is the
laptop-side end of the tunnel (localhost), not this machine's address.
"""

import json

from abliterated_vlm.config import Settings

PROVIDER_ID = "workstation"


def opencode_config(settings: Settings, local_port: int | None = None) -> dict:
    options = {"baseURL": f"http://127.0.0.1:{local_port or settings.port}/v1"}
    if settings.api_key:
        options["apiKey"] = settings.api_key
    model = {
        "name": "Huihui Qwen3.8 27B abliterated (Q4_K_L, workstation)",
        "tool_call": True,
        "reasoning": True,
        "attachment": settings.enable_vision,
        "modalities": {
            "input": ["text", "image"] if settings.enable_vision else ["text"],
            "output": ["text"],
        },
        "limit": {"context": settings.ctx_size, "output": min(16384, settings.ctx_size // 4)},
    }
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Workstation llama.cpp",
                "options": options,
                "models": {settings.model_alias: model},
            }
        },
    }


def render(settings: Settings, local_port: int | None = None) -> str:
    return json.dumps(opencode_config(settings, local_port), indent=2) + "\n"
