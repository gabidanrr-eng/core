"""Single source of truth for the Core Main version."""

__version__ = "0.1.0"

# Durable data-format versions. Bump when the corresponding serialized format changes
# incompatibly so importers and readers can detect and upgrade old data explicitly.
EXPORT_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1
SKILL_FORMAT_VERSION = 1
