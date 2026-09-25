"""Documentation research and web fetch: policy-governed, cached, untrusted-by-default external content.

``ResearchService`` (constructed lazily by ``CoreRuntime.extension("research")``) is the single entry
point for both the CLI (``core research docs|fetch``) and the model-facing ``docs_lookup`` /
``web_fetch`` tools.
"""

from __future__ import annotations

from coremain.research.errors import ResearchError
from coremain.research.service import ResearchService

__all__ = ["ResearchError", "ResearchService"]
