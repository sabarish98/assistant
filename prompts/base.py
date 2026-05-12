"""Pydantic models for prompt templates and metadata."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class PromptMetadata(BaseModel):
    """Metadata stored alongside every prompt version."""

    name: str = Field(..., description="Prompt identifier, e.g. 'qa_system'")
    version: str = Field(..., description="Semantic version string, e.g. 'v1'")
    description: str = Field(default="", description="What this prompt does")
    author: Optional[str] = Field(default=None, description="Who wrote this version")
    created_at: datetime = Field(default_factory=datetime.now)

    # Compatibility hints — useful when the workflow needs to know what
    # variables a prompt expects before rendering it.
    variables: List[str] = Field(
        default_factory=list,
        description="Template variable names expected by this prompt (e.g. ['context', 'question'])",
    )

    # Free-form notes — good place to record why a version was created
    changelog: str = Field(
        default="",
        description="What changed compared to the previous version",
    )

    @validator("version")
    def version_format(cls, v: str) -> str:
        """Enforce a simple vN format so sorting is unambiguous."""
        v = v.strip().lower()
        if not v.startswith("v"):
            raise ValueError(f"Version must start with 'v', got: {v!r}")
        suffix = v[1:]
        if not suffix.isdigit():
            raise ValueError(f"Version suffix must be an integer, got: {suffix!r}")
        return v

    @property
    def version_number(self) -> int:
        """Return the integer part of the version for comparison."""
        return int(self.version[1:])


class PromptTemplate(BaseModel):
    """A single versioned prompt — metadata + the raw template text."""

    metadata: PromptMetadata
    template: str = Field(..., min_length=1, description="The prompt text")

    # Optional: a dict of default variable values used when rendering
    defaults: Dict[str, Any] = Field(default_factory=dict)

    def render(self, **variables: Any) -> str:
        """
        Render the template by substituting {variable} placeholders.

        Falls back to defaults for any variable not explicitly provided.
        Raises KeyError if a required variable is missing from both
        the call and the defaults dict.

        Example
        -------
        >>> tpl.render(context="...", question="What is X?")
        """
        merged = {**self.defaults, **variables}
        try:
            return self.template.format(**merged)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Prompt '{self.metadata.name}@{self.metadata.version}' "
                f"requires variable '{missing}' but it was not provided."
            ) from exc

    @property
    def full_name(self) -> str:
        """Canonical identifier: 'name@version', e.g. 'qa_system@v1'."""
        return f"{self.metadata.name}@{self.metadata.version}"

    def __repr__(self) -> str:
        return f"PromptTemplate({self.full_name!r})"
