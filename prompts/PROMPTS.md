# Prompt Versioning Guide

How to create, iterate, and manage prompts in this project.

---

## How it works

Prompts live as YAML files under `prompts/templates/`. Each prompt has its own
subdirectory, and each version is a separate file. The `PromptRegistry` loads
them lazily and resolves references at runtime.

```
prompts/
├── base.py                        ← PromptTemplate + PromptMetadata models
├── registry.py                    ← PromptRegistry singleton
└── templates/
    └── qa_system/
        ├── v1.yaml                ← original version
        └── v2.yaml                ← next iteration (when you create it)
```

The workflow references a prompt by name — never by hardcoded string:

```python
# workflows/qa_workflow.py
_QA_SYSTEM_PROMPT_REF = "qa_system"          # always latest
# _QA_SYSTEM_PROMPT_REF = "qa_system@v1"     # pinned to v1
```

---

## YAML file schema

Every version file must follow this structure:

```yaml
name: qa_system          # must match the directory name
version: v2              # vN format — integer suffix only
description: >
  One-line summary of what this prompt does.
author: your-name
created_at: "2026-05-12T00:00:00"
variables: []            # list variable names if using {placeholders}
changelog: >
  What changed vs the previous version and why.

template: |
  Your prompt text goes here.
  It can span multiple lines.
  Use {variable_name} for dynamic substitution.

defaults: {}             # optional default values for variables
```

The only required keys are `name`, `version`, and `template`. Everything else
is optional but strongly recommended for traceability.

---

## Creating a new version

1. Copy the latest YAML in the prompt's directory:
   ```
   cp prompts/templates/qa_system/v1.yaml prompts/templates/qa_system/v2.yaml
   ```

2. Edit `v2.yaml` — bump `version` to `v2`, update `changelog`, modify `template`.

3. That's it. The registry picks up the new file automatically on the next run.
   No code changes needed as long as the workflow uses `"qa_system"` (latest).

---

## Pinning a version

To lock a workflow to a specific version — useful in production while you test
a new one — change the reference constant in the workflow file:

```python
# workflows/qa_workflow.py
_QA_SYSTEM_PROMPT_REF = "qa_system@v1"   # will never auto-upgrade
```

Change it back to `"qa_system"` when you're ready to adopt the latest.

---

## Using variable substitution

If your prompt needs dynamic content, declare the variables in the YAML and
use `{variable_name}` placeholders in the template:

```yaml
# prompts/templates/qa_system/v3.yaml
variables: [language, tone]
defaults:
  language: English
  tone: concise

template: |
  You are a research assistant. Answer in {language} using a {tone} tone.
  Use ONLY the context provided below.
```

Then render it in the workflow:

```python
system_prompt = prompt_registry.get("qa_system@v3").render(
    language="English",
    tone="detailed"
)
```

`render()` merges your kwargs with the YAML `defaults`, so you only need to
pass variables you want to override.

---

## Inspecting what's loaded

```python
from prompts.registry import prompt_registry

# List all prompt names
prompt_registry.list_prompts()
# → ['qa_system']

# List versions for a prompt
prompt_registry.list_versions("qa_system")
# → ['v1', 'v2']

# Inspect a specific version
tpl = prompt_registry.get("qa_system@v1")
print(tpl.metadata.changelog)
print(tpl.template)
```

---

## Reloading after editing a file

The registry caches templates in memory. If you edit a YAML while the process
is running (e.g. in a long-lived server), call `reload()` to pick up changes:

```python
prompt_registry.reload("qa_system")   # reload one prompt
prompt_registry.reload()              # reload everything
```

---

## Current prompts

| Name | Latest | Description |
|------|--------|-------------|
| `qa_system` | v1 | System prompt for the Q&A workflow — instructs the model to answer strictly from retrieved context |

Update this table whenever you add a new prompt or a significant new version.

---

## Conventions

- **One concern per prompt** — `qa_system` is for the system role only. If you
  add a human-turn template later, give it its own name (e.g. `qa_human`).
- **Always fill in `changelog`** — even a one-liner. It's the only record of
  why a version exists.
- **Don't delete old versions** — deprecate them by noting it in `changelog`.
  Old versions are cheap to keep and invaluable for debugging regressions.
- **Commit YAML files to git** — the diff history is your prompt audit trail.
