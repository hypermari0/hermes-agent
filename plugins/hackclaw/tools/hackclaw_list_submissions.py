"""hackclaw_list_submissions tool.

Returns the existing submissions for a hackathon. Used by the Strategist to
spot crowded angles and find a differentiation lane.
"""

from __future__ import annotations

from hackclaw.tools._runtime import as_json, run_sync, select_platform

NAME = "hackclaw_list_submissions"

SCHEMA = {
    "name": NAME,
    "description": (
        "List existing submissions for a hackathon. Returns name, teaser, "
        "and author for each. Used by the Strategist to find an "
        "under-served angle."
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
    submissions = await adapter.list_existing_submissions(hackathon_url)
    return {"submissions": [s.model_dump(mode="json") for s in submissions]}


def handle(params: dict) -> str:
    return as_json(run_sync(_run(params["hackathon_url"])))
