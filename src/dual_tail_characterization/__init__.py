"""Continuous PAM waveform simulation and clock-referenced eye extraction."""
from .config import Config
from .waveform import generate_waveform
from .eyes import extract_eye, paired_eyes

__all__ = ["Config", "generate_waveform", "extract_eye", "paired_eyes"]
