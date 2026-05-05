"""hackclaw_submit tool.

Flips a DRAFT project to ACTIVE on the hackathon platform. Called by the
orchestrator AFTER the user has approved the submission via a Hermes approval
prompt.

Never called from a subagent. The Submitter prepares the DRAFT; the
orchestrator handles the approval gate; this tool publishes.
"""

from __future__ import annotations

from hackclaw.platforms.taikai import TaikaiPlatform
from hackclaw.tools._runtime import as_json, run_sync, select_platform

NAME = "hackclaw_submit"

SCHEMA = {
    "name": NAME,
    "description": (
        "Flip a DRAFT project to ACTIVE on the hackathon platform. Should "
        "only be called by the orchestrator AFTER explicit user approval. "
        "Returns the new state and the public URL if the platform exposes "
        "one."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "hackathon_url": {
                "type": "string",
                "description": (
                    "Optional. Used to dispatch to the right platform "
                    "adapter. If omitted, the platform is inferred from "
                    "the project_id shape."
                ),
            },
        },
        "required": ["project_id"],
    },
}


def _looks_like_taikai_id(pid: str) -> bool:
    return bool(pid) and pid.islower() and pid.isalnum() and 20 <= len(pid) <= 30


async def _run(project_id: str, hackathon_url: str | None) -> dict:
    if hackathon_url:
        adapter, _kind = select_platform(hackathon_url)
    elif _looks_like_taikai_id(project_id):
        adapter = TaikaiPlatform()
    else:
        raise ValueError(
            f"Cannot infer platform from project_id={project_id!r}. "
            "Pass hackathon_url, or upgrade to v0.3 (browser adapter)."
        )

    result = await adapter.submit(project_id)
    return result.model_dump(mode="json")


def handle(params: dict) -> str:
    return as_json(
        run_sync(
            _run(
                params["project_id"],
                params.get("hackathon_url"),
            )
        )
    )
