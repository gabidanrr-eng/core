---
name: verification-before-completion
description: Before claiming work is done, run the checks that prove it and report their actual results. Use whenever you are about to submit a result or say something works.
license: MIT
metadata:
  core-kinds: bugfix, feature, refactor, tests, docs, architecture
  core-stage: implementation, correction, debugging
  core-triggers: done, complete, finished, works, fixed
  core-priority: "9"
  core-version: "1"
  core-inspired-by: obra/superpowers verification-before-completion
---

# Verification before completion

Core Main does not accept claims; it re-runs verification and gates completion on evidence.
Submitting "it works" without having observed it wastes a repair cycle and erodes trust.

Before calling `submit_result`:

1. **Re-read the request and acceptance criteria.** List each criterion and how you verified it.
2. **Run the project's real checks** that apply: tests (`run_tests`), and lint, type-check or
   build commands from `project_profile` when the change could affect them.
3. **Read the output.** "Command exited 0" is not enough when the output shows skipped or
   deselected tests, warnings turned errors, or zero tests collected.
4. **Inspect your own diff** (`git_diff`): no debug prints, commented-out code, unrelated edits,
   secrets, or changes to files outside the request.
5. **Report precisely:** commands run, pass/fail counts, what you could not verify and why.
   Use `remaining_issues` for anything unresolved. Confidence must match the evidence.

Forbidden phrases unless you observed the result: "should work", "tests pass", "fixed",
"no regressions".
