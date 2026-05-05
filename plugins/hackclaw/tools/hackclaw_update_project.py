"""hackclaw_update_project tool.

Fills in description, repo URL, deploy URL on a DRAFT project. Used by the
Submitter after the Storyteller has produced the project page HTML.

Platform dispatch: if `hackathon_url` is passed we use it; otherwise we infer
from project_id shape (TAIKAI cuids are lowercase alphanumeric, ~25 chars).
"""

from __future__ import annotations

from hackclaw.platforms.base import ProjectUpdates
from hackclaw.platforms.taikai import TaikaiPlatform
from hackclaw.tools._runtime import as_json, run_sync, select_platform

NAME = "hackclaw_update_project"

SCHEMA = {
    "name": NAME,
    "description": (
        "Update a DRAFT project on the hackathon platform with description "
        "HTML, repo URL, and deploy URL. Used by the Submitter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "description_html": {"type": "string"},
            "repo_url": {"type": "string"},
            "deploy_url": {"type": "string"},
            "hackathon_url": {
                "type": "string",
                "description": (
                    "Optional. Used to dispatch to the right platform "
                    "adapter. If omitted, the platform is inferred from "
                    "the project_id shape."
                ),
            },
        },
        "required": ["project_id", "description_html"],
    },
}


def _looks_like_taikai_id(pid: str) -> bool:
    return bool(pid) and pid.islower() and pid.isalnum() and 20 <= len(pid) <= 30


async def _run(
    project_id: str,
    description_html: str,
    repo_url: str | None,
    deploy_url: str | None,
    hackathon_url: str | None,
) -> dict:
    if hackathon_url:
        adapter, _kind = select_platform(hackathon_url)
    elif _looks_like_taikai_id(project_id):
        adapter = TaikaiPlatform()
    else:
        raise ValueError(
            f"Cannot infer platform from project_id={project_id!r}. "
            "Pass hackathon_url, or upgrade to v0.3 (browser adapter)."
        )

    updates = ProjectUpdates(
        description_html=description_html,
        repo_url=repo_url,
        deploy_url=deploy_url,
    )
    await adapter.update_project(project_id, updates)
    return {"project_id": project_id, "state": "DRAFT", "updated": True}


def handle(params: dict) -> str:
    return as_json(
        run_sync(
            _run(
                params["project_id"],
                params["description_html"],
                params.get("repo_url"),
                params.get("deploy_url"),
                params.get("hackathon_url"),
            )
        )
    )
