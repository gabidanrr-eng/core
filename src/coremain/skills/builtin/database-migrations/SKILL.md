---
name: database-migrations
description: Write safe schema and data migrations - forward-only, idempotent where possible, tested against a copy, with separate backfills. Use when changing database schemas or data.
license: MIT
metadata:
  core-stage: planning, implementation, correction, review
  core-triggers: migration, migrate, schema, alembic, column, table, index, sqlite, postgres, mysql, backfill
  core-priority: "6"
  core-version: "1"
---

# Database migrations

- **Use the project's migration tool** (Alembic, Django, Prisma, Rails, a custom runner) and its
  naming/ordering conventions. Never edit a migration that may already have run elsewhere; add a
  new one.
- **Separate schema from data**: add nullable columns first, backfill in a separate step, then
  add constraints. Large backfills should be batched.
- **Be idempotent and ordered**: guard with the tool's version tracking; migrations must apply
  cleanly to a fresh database *and* to the current production shape.
- **Mind locking and compatibility**: adding indexes or rewriting tables can lock; code must
  work with both the old and new schema during a rolling deploy.
- **Downgrades**: provide them when the tool supports it and data loss is not implied; otherwise
  say explicitly that the migration is irreversible.
- **Test**: apply to an empty database and to a fixture with realistic data; run the app's tests;
  verify the resulting schema (columns, types, constraints, indexes).
