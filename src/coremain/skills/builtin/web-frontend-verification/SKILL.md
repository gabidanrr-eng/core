---
name: web-frontend-verification
description: Verify user-interface changes by actually rendering them - run the page, inspect the accessibility snapshot, check console errors and take a screenshot. Use for HTML/CSS/JS/React/Vue/Svelte work.
license: MIT
metadata:
  core-stage: implementation, correction, review
  core-triggers: ui, page, button, form, css, html, frontend, layout, component, react, vue, svelte, browser, style
  core-languages: javascript, typescript, html, css
  core-frameworks: react, vue, svelte, next, nuxt, angular, vite
  core-priority: "6"
  core-version: "1"
  core-inspired-by: microsoft/playwright-mcp (snapshot-first interaction)
---

# Web frontend verification

Unit tests rarely prove a UI works. When browser tools are available:

1. Start the dev server or open the static file (relative paths resolve inside the workspace).
   Use the project's own script (`npm run dev`, `vite`, …) via `run_command` with a timeout.
2. `browser_navigate` to the page, then `browser_snapshot` — reason about the accessibility tree
   (roles, names, states) rather than pixels. Interact with `browser_click`/`browser_type`
   using the snapshot's element refs.
3. `browser_console` after interactions: uncaught errors or failed requests mean it is broken
   even if it looks right.
4. `browser_screenshot` for visual changes; describe what you verified, not just that you took it.
5. Check basics: keyboard reachability, labels on inputs, visible focus, no layout overflow at a
   narrow width if the change affects layout.

Also run the project's frontend tests and linters. If browser tools are unavailable, say that
the rendering was not verified.
