"""HackathonPlatform protocol. Uniform interface every platform adapter implements."""

from typing import Protocol

from pydantic import BaseModel


class Brief(BaseModel):
    """Normalized hackathon brief."""

    title: str
    short_description: str
    full_description: str | None = None
    theme_tags: list[str] = []
    timeline: dict[str, str] = {}
    prizes: list[dict] = []
    rules: list[str] = []
    raw_url: str


class Submission(BaseModel):
    """An existing submission scouted from the hackathon."""

    name: str
    teaser: str
    author: str | None = None
    angle_summary: str | None = None


class ProjectDraft(BaseModel):
    name: str
    teaser: str


class ProjectUpdates(BaseModel):
    description_html: str | None = None
    repo_url: str | None = None
    deploy_url: str | None = None
    video_url: str | None = None


class SubmissionResult(BaseModel):
    project_id: str
    state: str
    public_url: str | None = None


class HackathonPlatform(Protocol):
    """Every adapter implements these five methods."""

    async def get_brief(self, hackathon_url: str) -> Brief: ...
    async def list_existing_submissions(self, hackathon_url: str) -> list[Submission]: ...
    async def create_draft_project(self, hackathon_url: str, draft: ProjectDraft) -> str: ...
    async def update_project(self, project_id: str, updates: ProjectUpdates) -> None: ...
    async def submit(self, project_id: str) -> SubmissionResult: ...
