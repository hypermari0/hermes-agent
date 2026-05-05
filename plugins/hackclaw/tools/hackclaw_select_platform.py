"""hackclaw_select_platform tool.

Returns "taikai" or "browser" for a given hackathon URL. The squad calls this
first to know which adapter is in play.
"""

from __future__ import annotations

from hackclaw.tools._runtime import as_json, select_platform_kind

NAME = "hackclaw_select_platform"

SCHEMA = {
    "name": NAME,
    "description": (
        "Pick the right adapter for a hackathon URL. Returns 'taikai' for "
        "TAIKAI hackathons or 'browser' for everything else (handled via "
        "Claude in Chrome MCP in v0.3+)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hackathon_url": {
                "type": "string",
                "description": (
                    "Full URL of the hackathon "
                    "(e.g. https://taikai.network/<org>/hackathons/<slug>)"
                ),
            },
        },
        "required": ["hackathon_url"],
    },
}


def handle(params: dict) -> str:
    url = params["hackathon_url"]
    return as_json({"platform": select_platform_kind(url)})
