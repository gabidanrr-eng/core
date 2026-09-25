"""Typed, layered, inspectable configuration."""

from coremain.config.loader import EffectiveConfig, load_config
from coremain.config.schema import CoreConfig

__all__ = ["CoreConfig", "EffectiveConfig", "load_config"]
