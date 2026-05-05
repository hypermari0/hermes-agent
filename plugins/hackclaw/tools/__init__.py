"""Hermes tool wrappers for HackClaw.

Each module exports a `register(registry)` callable that Hermes invokes at
plugin load time. The exact signature depends on the Hermes plugin API
version. The pattern below targets upstream Hermes plugin API >= 1.0.0.
"""
