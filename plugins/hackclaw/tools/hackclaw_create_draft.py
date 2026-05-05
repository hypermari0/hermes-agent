"""hackclaw_create_draft tool.

Creates a DRAFT project on the platform. Returns the project ID.

Recursive-demo short-circuit: if HACKCLAW_TARGET_PROJECT_ID is set in the
environment, this tool returns that pre-existing project_id without creating
a new one. The Submitter then routes straight into hackclaw_update_project.
"""

from __future__ import annotations

import os

from hackclaw.platforms.base import ProjectDraft
from hackclaw.tools._runtime import as_json, run_sync, select_platform

NAME = "hackclaw_create_draft"

SCHEMA = {
    "name": NAME,
    "description": (
        "Create a DRAFT project on the hackathon platform. Returns the "
        "platform project ID. If HACKCLAW_TARGET_PROJECT_ID is set, returns "
        "that ID without creating (used for the recursive demo). Used by "
        "the Submitter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hackathon_url": {"type": "string"},
            "name": {"type": "string", "description": "Project name (3-80 chars)"},
            "teaser": {"type": "string", "description": "Short tagline (3-200 chars)"},
        },
        "required": ["hackathon_url", "name", "teaser"],
    },
}


async def _run(hackathon_url: str, name: str, teaser: str) -> dict:
    target = os.environ.get("HACKCLAW_TARGET_PROJECT_ID")
    if target:
        return {
            "project_id": target,
            "state": "DRAFT",
            "reused_existing": True,
            "source": "HACKCLAW_TARGET_PROJECT_ID",
        }

    adapter, _kind = select_platform(hackathon_url)
    draft = ProjectDraft(name=name[:80], teaser=teaser[:200])
    project_id = await adapter.create_draft_project(hackathon_url, draft)
    return {"project_id": project_id, "state": "DRAFT", "reused_existing": False}


def handle(params: dict) -> str:
    return as_json(
        run_sync(
            _run(
                params["hackathon_url"],
                params["name"],
                params["teaser"],
            )
        )
    )
