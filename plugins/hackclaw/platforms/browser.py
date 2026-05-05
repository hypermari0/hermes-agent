"""Browser platform adapter (v0.3+).

Drives non-TAIKAI platforms (Devpost, ETHGlobal, Encode, Dorahacks, Akindo,
custom hackathon sites) via Claude in Chrome MCP. Stub for v0.2.
"""

from hackclaw.platforms.base import (
    Brief,
    HackathonPlatform,
    ProjectDraft,
    ProjectUpdates,
    Submission,
    SubmissionResult,
)


class BrowserPlatform(HackathonPlatform):
    """Stub. v0.3 wires Claude in Chrome MCP."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "BrowserPlatform is not implemented in v0.2. "
            "v0.3 will wire Claude in Chrome MCP for non-TAIKAI hackathons."
        )

    async def get_brief(self, hackathon_url: str) -> Brief:
        raise NotImplementedError

    async def list_existing_submissions(self, hackathon_url: str) -> list[Submission]:
        raise NotImplementedError

    async def create_draft_project(self, hackathon_url: str, draft: ProjectDraft) -> str:
        raise NotImplementedError

    async def update_project(self, project_id: str, updates: ProjectUpdates) -> None:
        raise NotImplementedError

    async def submit(self, project_id: str) -> SubmissionResult:
        raise NotImplementedError
