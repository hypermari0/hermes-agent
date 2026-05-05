"""Shared runtime helpers for HackClaw tools.

Every tool dispatches to a HackathonPlatform adapter. Selection is URL-based:
TAIKAI URLs go to TaikaiPlatform; everything else routes to BrowserPlatform.

This module also provides a small async-to-sync bridge. Hermes plugin tool
handlers are sync-callable (per the upstream plugin authoring guide), so each
tool exposes a sync `handle(params)` that calls into our async adapters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable

from hackclaw.platforms.base import HackathonPlatform
from hackclaw.platforms.browser import BrowserPlatform
from hackclaw.platforms.taikai import TaikaiPlatform


def select_platform(hackathon_url: str) -> tuple[HackathonPlatform, str]:
    """Return (adapter, kind) for a hackathon URL."""
    if "taikai.network" in hackathon_url:
        return TaikaiPlatform(), "taikai"
    return BrowserPlatform(), "browser"


def select_platform_kind(hackathon_url: str) -> str:
    """Return just the kind without instantiating the adapter."""
    if "taikai.network" in hackathon_url:
        return "taikai"
    return "browser"


def run_sync(coro: Awaitable[Any]) -> Any:
    """Run an async coroutine from a sync tool handler.

    Hermes plugin tool handlers are sync. Our adapters are async. Bridge them
    by spinning up a fresh event loop when needed. If we're already inside a
    running loop (uncommon for plugin handlers but possible if Hermes ever
    invokes us from one), fall back to a worker-thread loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def as_json(payload: Any) -> str:
    """Serialize a tool result for return to Hermes.

    Upstream tool handlers return strings. We dump the structured payload as
    JSON so the LLM gets a well-formed object back.
    """
    return json.dumps(payload, ensure_ascii=False, default=str)
