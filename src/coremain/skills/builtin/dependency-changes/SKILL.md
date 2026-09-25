---
name: dependency-changes
description: Add, upgrade or remove dependencies conservatively - respect lockfiles and installed versions, read the changelog, and verify the build and tests. Use when a task involves packages or versions.
license: MIT
metadata:
  core-stage: planning, implementation, correction
  core-triggers: dependency, dependencies, upgrade, downgrade, install, package, version, bump, requirements, lockfile, npm, pip, cargo
  core-priority: "5"
  core-version: "1"
---

# Dependency changes

- **Prefer what is installed.** Check the lockfile and manifest for the version actually in use
  and write code for that version; look up its API with `docs_lookup` rather than memory.
- **Use the project's package manager** (detect from lockfiles: `uv.lock`, `poetry.lock`,
  `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock`, `Cargo.lock`). Never mix managers.
- **Minimal changes**: upgrade only what the task needs; no drive-by upgrades. Pin or constrain
  consistently with the existing style.
- **Read the changelog** for breaking changes between the old and new version before upgrading.
- Installing packages needs network and may require approval; if denied, explain what is needed.
- **Verify**: lockfile updated consistently, the build and full test suite pass, and no new
  deprecation warnings you introduced.
