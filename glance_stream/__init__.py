"""Glance-Stream: standing yes/no rules and boxes over a live video stream with one small video-language model."""
from glance_stream.rules import Rule, Trigger, violations
from glance_stream.api import GlanceStream, Monitor, Readout, Box

__version__ = "0.1.0"
__all__ = ["GlanceStream", "Monitor", "Readout", "Box", "Rule", "Trigger", "violations"]
