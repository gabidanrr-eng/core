---
name: writing-documentation
description: Write or update documentation that matches the code's real behaviour, with runnable examples and no invented features. Use for README, docs and docstring tasks.
license: MIT
metadata:
  core-kinds: docs
  core-stage: implementation, correction, review
  core-triggers: document, documentation, readme, docs, docstring, guide, changelog, usage
  core-priority: "6"
  core-version: "1"
---

# Writing documentation

- **Derive every statement from the code** you read in this task: flags, defaults, environment
  variables, error messages. Never document planned or assumed behaviour.
- **Lead with the task a reader wants to do**, then the details. Short sections, concrete
  examples, copy-pasteable commands.
- **Run the examples** when feasible (`run_command`) and show real output shapes.
- **Match the existing docs** (tone, heading levels, terminology) and update every place that
  the change makes stale (README, docs pages, help text, changelog).
- Keep it truthful about limitations and requirements (versions, platforms, credentials).
