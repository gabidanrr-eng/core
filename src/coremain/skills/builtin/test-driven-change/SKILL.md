---
name: test-driven-change
description: Make behaviour changes test-first - write a failing test, see it fail for the right reason, implement the minimum, then run the suite. Use when adding features, fixing bugs or writing tests.
license: MIT
metadata:
  core-kinds: feature, bugfix, tests
  core-stage: implementation, correction, debugging
  core-triggers: test, tests, tdd, coverage, unit test, add a feature, implement
  core-priority: "6"
  core-version: "1"
  core-inspired-by: obra/superpowers test-driven-development
---

# Test-driven change

1. **Find the existing test conventions** (framework, file layout, fixtures, naming) with
   `glob_files`, `project_profile` and one or two nearby tests. Follow them exactly.
2. **Write the failing test first.** It should describe the behaviour a user would notice,
   not the implementation. One behaviour per test; clear name.
3. **Run it and watch it fail** for the expected reason (assertion on the new behaviour, not an
   import error or typo). A test that passes before the change proves nothing.
4. **Implement the minimum** that makes it pass. Resist adding untested options.
5. **Run the whole relevant suite**, not just the new test. Fix regressions you introduced.
6. **Refactor** only with the suite green, then run it again.

## Rules
- Never delete, skip, `xfail` or weaken an existing test to make the suite pass. If a test is
  genuinely wrong, explain why in `remaining_issues` and leave the decision to the user.
- Cover the edge cases the change introduces (empty input, boundaries, error paths).
- Tests must be deterministic: no sleeps, real network, wall-clock or random order assumptions.
- Report the exact test command and its result in `submit_result`.
