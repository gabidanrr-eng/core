"""Browser automation: policy-governed Playwright sessions, snapshot-first tools and declarative checks.

Enforcement is layered. Tool calls go through the policy engine like every other tool; every
request a page makes is checked against ``browser.allowed_origins`` by a per-context route
handler (which also confines ``file://`` loads to the task workspace); and every network
connection, including redirect hops and WebSockets that route interception cannot see, goes
through a per-session loopback egress proxy that applies the same origin policy.
"""

from __future__ import annotations

from coremain.browser.origins import FileScope, OriginPolicy, Verdict

__all__ = ["FileScope", "OriginPolicy", "Verdict"]
