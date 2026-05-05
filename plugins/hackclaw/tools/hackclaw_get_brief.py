"""hackclaw_get_brief tool.

Loads and normalizes a hackathon brief from the platform.
"""

from __future__ import annotations

from hackclaw.tools._runtime import as_json, run_sync, select_platform

NAME = "hackclaw_get_brief"

SCHEMA = {
    "name": NAME,
    "description": (
        "Load and normalize a hackathon brief from the platform. Returns "
        "title, short_description, full_description, theme_tags, timeline, "
        "prizes, rules. Used by the Strategist to understand what to build."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hackathon_url": {"type": "string"},
        },
        "required": ["hackathon_url"],
    },
}


async def _run(hackathon_url: str) -> dict:
    adapter, _kind = select_platform(hackathon_url)
    brief = await adapter.get_brief(hackathon_url)
    return brief.model_dump(mode="json")


def handle(params: dict) -> str:
    return as_json(run_sync(_run(params["hackathon_url"])))
