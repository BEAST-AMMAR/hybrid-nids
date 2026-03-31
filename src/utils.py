"""
Shared utilities for Hybrid NIDS.
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Return the repo root (one level above src/)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load config/config.yaml and return as a dict."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return get_project_root() / relative
