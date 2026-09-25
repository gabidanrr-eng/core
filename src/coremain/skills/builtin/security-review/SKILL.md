---
name: security-review
description: Check code that touches input handling, authentication, secrets, queries, files or network for common vulnerabilities and fix them safely. Use for security-sensitive changes and security reviews.
license: MIT
metadata:
  core-kinds: review, bugfix, feature
  core-stage: review, implementation, correction
  core-triggers: security, auth, login, password, token, secret, credential, sql, injection, xss, csrf, permission, upload, jwt, oauth, crypto
  core-priority: "7"
  core-version: "1"
---

# Security review

Check every place where untrusted data crosses a boundary.

- **Injection**: SQL built with string formatting (use parameters), shell commands with user
  input (use argv lists, never `shell=True` with interpolation), template injection, path
  traversal (normalize and confine paths), unsafe deserialization (`pickle`, `yaml.load`).
- **Secrets**: never hardcode credentials, keys or tokens; read them from the environment or a
  secret store, never log or echo them, and keep them out of tests and fixtures.
- **AuthN/AuthZ**: every sensitive operation checks *who* and *whether allowed*, server-side;
  object-level access checks (IDOR); constant-time comparison for secrets.
- **Web**: output encoding (XSS), CSRF protection on state-changing requests, safe redirects,
  CORS not wildcarded with credentials, security headers where the framework expects them.
- **SSRF / outbound calls**: validate destinations, restrict schemes and internal addresses.
- **Crypto**: use vetted libraries and modern algorithms; no homemade crypto, no MD5/SHA1 for
  passwords (use bcrypt/scrypt/argon2), secure random for tokens.
- **Dependencies**: prefer maintained packages; do not add dependencies casually.

Report each issue with the exploit scenario, evidence and a concrete fix. Fixes must not break
legitimate behaviour — add tests for both the attack and the normal path.
