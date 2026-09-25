---
name: systematic-debugging
description: Find and fix the root cause of a bug, failing test, crash or regression by reproducing it first and testing one hypothesis at a time. Use for any failure investigation.
license: MIT
metadata:
  core-kinds: bugfix
  core-stage: debugging, correction
  core-triggers: failing, fails, error, exception, traceback, crash, bug, regression, broken, flaky
  core-priority: "8"
  core-version: "1"
  core-inspired-by: obra/superpowers systematic-debugging
---

# Systematic debugging

Never fix what you have not reproduced. Guessing produces symptom patches that pass one test
and break three others.

## 1. Reproduce
- Run the failing test or command and read the **entire** error, including the first frame in
  project code, not just the last line.
- If there is no reproduction yet, write the smallest failing test that shows the bug. It
  becomes the regression test.
- If it does not reproduce, say so and investigate the environment (versions, env vars, data)
  instead of changing code.

## 2. Locate
- Trace the bad value backwards: where was it produced, transformed, and consumed?
- Use `find_symbol`, `find_references`, `file_outline` and `git_log`/`git_blame` for recent
  changes near the failure. Recent changes are prime suspects.
- Compare with a working path (another caller, an older commit, a similar function).

## 3. Hypothesize and test — one at a time
- State one hypothesis with `record_hypothesis` ("X is null because Y skips initialization").
- Design the smallest experiment that could refute it (a print, a narrower test, a REPL check)
  and record the outcome with `record_experiment`.
- Change one thing per experiment. If three hypotheses fail, stop and question your model of
  the code (architecture, assumptions about inputs) before trying a fourth patch.

## 4. Fix the cause
- Fix where the defect originates, not where it surfaces. Avoid catch-all `try/except`,
  special-casing the failing input, or loosening assertions.
- Keep the fix minimal; note unrelated problems in `remaining_issues` instead of fixing them.

## 5. Prove it
- The reproduction now passes, and the full relevant test suite still passes.
- Report exactly what you ran and the results. Never claim a fix you did not observe.
