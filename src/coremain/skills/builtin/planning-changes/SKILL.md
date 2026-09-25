---
name: planning-changes
description: Turn a feature, refactor or architecture request into a small, verifiable plan grounded in the actual codebase. Use at planning or architecture stages and for multi-step work.
license: MIT
metadata:
  core-kinds: feature, refactor, architecture
  core-stage: planning, architecture
  core-triggers: plan, design, architecture, migrate, restructure, multi-step, roadmap
  core-priority: "7"
  core-version: "1"
  core-inspired-by: obra/superpowers writing-plans
---

# Planning changes

A plan is only useful if another engineer could execute and verify it without guessing.

## Understand before deciding
- Map the relevant area: `repo_map` (focused), `find_symbol`, `impact_analysis` on the files you
  expect to touch, and `memory_search` for recorded conventions and decisions.
- Identify existing patterns to reuse (similar endpoints, helpers, test fixtures). New
  abstractions need a reason the existing ones cannot serve.

## Write the plan (`submit_plan`)
- **approach**: the chosen design in a few sentences and why it beats the obvious alternative.
- **steps**: small, ordered, each independently verifiable ("add X to Y with test Z"), naming
  concrete files and functions.
- **risks**: what could break (callers, data, performance, compatibility) and how you will know.
- **test_strategy**: which tests prove each acceptance criterion; existing suites to run.
- **acceptance**: observable criteria, not activities.
- **decisions**: record genuine trade-offs with `record_decision` so they persist.

## Keep scope honest
- Do the smallest change that satisfies the request. List follow-ups instead of doing them.
- If requirements are ambiguous in a way that changes the design, ask (`ask_user`) once with
  concrete options rather than guessing.
