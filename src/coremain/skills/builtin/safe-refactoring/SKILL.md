---
name: safe-refactoring
description: Restructure code without changing behaviour - rename, extract, move or simplify in small steps with tests green before and after. Use for refactoring requests.
license: MIT
metadata:
  core-kinds: refactor
  core-stage: planning, implementation, correction
  core-triggers: refactor, rename, extract, move, restructure, clean up, simplify, deduplicate
  core-priority: "7"
  core-version: "1"
---

# Safe refactoring

1. **Establish a baseline**: run the tests before touching anything. If they fail already,
   record that; do not mix fixing them into the refactor unless asked.
2. **Find every reference** with `find_references` and `search_text` (including strings,
   configuration, docs, dynamic imports and tests). `impact_analysis` shows dependants.
3. **Change in small steps**, running the relevant tests after each. Prefer mechanical edits
   (`edit_file` with `replace_all` only when every match is intended).
4. **Preserve public interfaces** unless the request says otherwise; if you must change one, keep
   a compatibility alias or document the break explicitly.
5. **No behaviour changes.** Refactors that also "fix" things are hard to review; list such
   findings in `remaining_issues`.
6. **Finish with the full suite** and a scan for leftovers of the old name/structure.
