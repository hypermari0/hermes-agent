"""HackClaw Hermes plugin.

Hermes loads this plugin from ~/.hermes/plugins/hackclaw/ (symlinked by
install.sh). On startup, Hermes calls `register(ctx)` and we wire each tool
schema to its sync handler via `ctx.register_tool(name, schema, handler)`.

Tools live under hackclaw.tools. Platform adapters live under
hackclaw.platforms. The TAIKAI adapter has two routes (MCP via Hermes, or
direct GraphQL) selected by HACKCLAW_TAIKAI_VIA.
"""

from __future__ import annotations

from hackclaw.tools import (
    hackclaw_create_draft,
    hackclaw_get_brief,
    hackclaw_list_submissions,
    hackclaw_select_platform,
    hackclaw_submit,
    hackclaw_update_project,
)

__version__ = "0.2.0"

_TOOLS = (
    hackclaw_select_platform,
    hackclaw_get_brief,
    hackclaw_list_submissions,
    hackclaw_create_draft,
    hackclaw_update_project,
    hackclaw_submit,
)


def register(ctx) -> None:
    """Register all HackClaw tools. Called once by the Hermes plugin loader.

    Mirrors the upstream plugin pattern (see plugins/spotify/__init__.py in
    NousResearch/hermes-agent). Each tool module exposes NAME, SCHEMA, and
    handle(params) -> str.
    """
    for tool in _TOOLS:
        ctx.register_tool(tool.NAME, tool.SCHEMA, tool.handle)
