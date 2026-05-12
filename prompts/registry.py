"""
PromptRegistry — loads, caches, and resolves versioned prompt templates.

Resolution rules
----------------
- "qa_system"      → latest version of the 'qa_system' prompt
- "qa_system@v1"   → exactly version v1
- "qa_system@v2"   → exactly version v2

Templates are loaded lazily from YAML files under prompts/templates/.
Each prompt name maps to a sub-directory; each version is one YAML file.

Directory layout expected
-------------------------
prompts/templates/
└── <prompt_name>/
    ├── v1.yaml
    └── v2.yaml

YAML file schema
----------------
name: qa_system
version: v1
description: "..."
author: "..."
created_at: "2026-05-12T00:00:00"
variables: [context, question]
changelog: "Initial version"
template: |
  You are a research assistant...
defaults: {}          # optional
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from core.logger import app_logger
from prompts.base import PromptMetadata, PromptTemplate

# Root of the templates directory, relative to this file
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Regex for the "name@version" reference syntax
_REF_RE = re.compile(r"^(?P<name>[a-z0-9_]+)(?:@(?P<version>v\d+))?$")


class PromptRegistry:
    """
    Central store for all versioned prompt templates.

    Usage
    -----
    >>> registry = PromptRegistry()
    >>> tpl = registry.get("qa_system")          # latest
    >>> tpl = registry.get("qa_system@v1")       # pinned
    >>> rendered = tpl.render(context="...", question="...")
    """

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        self._templates_dir = templates_dir or _TEMPLATES_DIR
        # Cache: (name, version) → PromptTemplate
        self._cache: Dict[Tuple[str, str], PromptTemplate] = {}
        # Tracks which names have been fully scanned from disk
        self._scanned: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, ref: str) -> PromptTemplate:
        """
        Resolve a prompt reference and return the PromptTemplate.

        Parameters
        ----------
        ref : str
            Either "prompt_name" (latest) or "prompt_name@vN" (pinned).

        Raises
        ------
        KeyError  if the prompt name or version does not exist.
        """
        name, version = self._parse_ref(ref)
        self._ensure_loaded(name)

        if version is None:
            version = self._latest_version(name)

        key = (name, version)
        if key not in self._cache:
            raise KeyError(
                f"Prompt '{name}@{version}' not found. "
                f"Available versions: {self.list_versions(name)}"
            )

        return self._cache[key]

    def list_versions(self, name: str) -> List[str]:
        """Return all known versions for a prompt name, sorted ascending."""
        self._ensure_loaded(name)
        versions = [v for (n, v) in self._cache if n == name]
        return sorted(versions, key=lambda v: int(v[1:]))

    def list_prompts(self) -> List[str]:
        """Return all prompt names discovered in the templates directory."""
        if not self._templates_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._templates_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )

    def reload(self, name: Optional[str] = None) -> None:
        """
        Invalidate the cache and reload from disk.

        Pass a name to reload only that prompt; omit to reload everything.
        """
        if name:
            keys_to_drop = [k for k in self._cache if k[0] == name]
            for k in keys_to_drop:
                del self._cache[k]
            self._scanned.discard(name)
            self._ensure_loaded(name)
        else:
            self._cache.clear()
            self._scanned.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_ref(self, ref: str) -> Tuple[str, Optional[str]]:
        """Parse "name" or "name@vN" into (name, version_or_None)."""
        m = _REF_RE.match(ref.strip().lower())
        if not m:
            raise ValueError(
                f"Invalid prompt reference {ref!r}. "
                "Expected 'prompt_name' or 'prompt_name@vN'."
            )
        return m.group("name"), m.group("version")

    def _ensure_loaded(self, name: str) -> None:
        """Load all versions of *name* from disk if not already cached."""
        if name in self._scanned:
            return

        prompt_dir = self._templates_dir / name
        if not prompt_dir.exists():
            raise KeyError(
                f"No prompt directory found for '{name}' "
                f"(looked in {prompt_dir})."
            )

        yaml_files = sorted(prompt_dir.glob("v*.yaml"))
        if not yaml_files:
            raise KeyError(
                f"Prompt directory '{prompt_dir}' exists but contains no v*.yaml files."
            )

        for yaml_path in yaml_files:
            tpl = self._load_yaml(yaml_path)
            key = (tpl.metadata.name, tpl.metadata.version)
            self._cache[key] = tpl
            app_logger.debug(f"[PromptRegistry] loaded {tpl.full_name} from {yaml_path}")

        self._scanned.add(name)
        app_logger.info(
            f"[PromptRegistry] loaded {len(yaml_files)} version(s) for '{name}'"
        )

    def _latest_version(self, name: str) -> str:
        """Return the highest version string for a given prompt name."""
        versions = self.list_versions(name)
        if not versions:
            raise KeyError(f"No versions found for prompt '{name}'")
        # list_versions returns sorted ascending; last = latest
        return versions[-1]

    @staticmethod
    def _load_yaml(path: Path) -> PromptTemplate:
        """Parse a single YAML file into a PromptTemplate."""
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            raise ValueError(f"Failed to parse prompt YAML at {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Prompt YAML at {path} must be a mapping, got {type(data)}")

        # Split top-level keys into metadata vs template/defaults
        template_text = data.pop("template", None)
        defaults = data.pop("defaults", {})

        if not template_text:
            raise ValueError(f"Prompt YAML at {path} is missing the 'template' key")

        metadata = PromptMetadata(**data)

        return PromptTemplate(
            metadata=metadata,
            template=template_text,
            defaults=defaults or {},
        )


# ---------------------------------------------------------------------------
# Global singleton — import this everywhere
# ---------------------------------------------------------------------------
prompt_registry = PromptRegistry()

__all__ = ["PromptRegistry", "prompt_registry"]
