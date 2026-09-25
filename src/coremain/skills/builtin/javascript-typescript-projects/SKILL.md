---
name: javascript-typescript-projects
description: Conventions for JavaScript and TypeScript repositories - package managers, scripts, test runners, module systems and type safety. Use when the project is JS/TS.
license: MIT
metadata:
  core-languages: javascript, typescript
  core-stage: implementation, correction, debugging, review
  core-triggers: node, npm, pnpm, yarn, typescript, javascript, jest, vitest, eslint, tsconfig
  core-priority: "4"
  core-version: "1"
---

# JavaScript and TypeScript projects

- **Package manager** from the lockfile: `pnpm-lock.yaml` → pnpm, `yarn.lock` → yarn,
  `package-lock.json` → npm, `bun.lockb`/`bun.lock` → bun. Run scripts through it
  (`pnpm test`, `npm run build`); read `package.json` scripts before inventing commands.
- **Tests**: use the configured runner (Vitest, Jest, `node --test`, Playwright). Run one file
  first, then the suite. Avoid watch mode in automation (`--run`, `CI=1`).
- **Modules**: respect `"type": "module"` vs CommonJS, file extensions in ESM imports, and path
  aliases from `tsconfig.json`/bundler config.
- **Types**: keep `strict` guarantees; no `any`/`@ts-ignore` to silence real errors. Run `tsc
  --noEmit` (or the project's typecheck script) when you change types.
- **Pitfalls**: unhandled promise rejections, `==` vs `===`, mutating props/state, stale closures
  in React hooks, floating-point money arithmetic (use integer cents).
