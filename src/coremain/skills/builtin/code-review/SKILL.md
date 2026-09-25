---
name: code-review
description: Review a change for correctness, security, tests and requirement fit with evidence-backed, calibrated findings. Use in review stages and when asked to review code or a diff.
license: MIT
metadata:
  core-kinds: review
  core-stage: review
  core-triggers: review, audit, check my changes, pull request, pr
  core-priority: "8"
  core-version: "1"
  core-inspired-by: obra/superpowers requesting-code-review
---

# Code review

Your job is to find real problems, not to restate the diff or praise it.

## Method
1. Read the request/acceptance criteria first, then the diff, then the surrounding code the diff
   depends on (callers, tests, data models). Do not review the diff in isolation.
2. Verify instead of assuming: open the files, check how functions are called, and run tests
   (`run_tests`) when a claim depends on behaviour.
3. Walk the risky paths: error handling, empty/huge inputs, concurrency, resource cleanup,
   backwards compatibility, security boundaries.

## Findings
- Each finding needs a location (`file`, `line`), a concrete rationale, evidence (quoted code or
  test output) and a specific remediation.
- Severity: **critical/high** = wrong results, data loss, security hole, broken build or tests;
  **medium** = likely bug in a less common path, missing test for new behaviour; **low/info** =
  maintainability. Do not inflate style preferences.
- Flag weakened, skipped or deleted tests, and claims in the summary that the diff does not
  support (`misleading_claim`).

## Verdict
- `approve` only when no critical/high issue remains and the criteria are met.
- `request_changes` for any blocking issue. `inconclusive` if you could not verify something
  essential — say what is missing.
