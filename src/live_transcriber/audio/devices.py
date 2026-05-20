"""
Audio device detection and interactive selection helpers.
"""

from __future__ import annotations

import logging
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


def find_blackhole_device() -> tuple[Optional[int], Optional[str]]:
    """Auto-detect the first BlackHole input device available on macOS."""
    for idx, dev in enumerate(sd.query_devices()):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx, dev["name"]
    return None, None


def list_input_devices() -> list[dict]:
    """Return a list of available input devices as dicts with 'index' and 'name'."""
    return [
        {"index": idx, "name": dev["name"]}
        for idx, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0
    ]


def print_input_devices() -> None:
    """Print all available input devices to stdout."""
    print("\nAvailable Input Devices:\n")
    for entry in list_input_devices():
        print(f"  {entry['index']}: {entry['name']}")
    print()


def find_default_input_device() -> tuple[int, str]:
    """
    Return the OS-level default input device (index, name).

    On macOS this is typically the built-in MacBook Pro Microphone.
    Raises ``RuntimeError`` if no default input device is configured.
    """
    default_input_idx = sd.default.device[0]
    if default_input_idx < 0:
        raise RuntimeError(
            "No default input device found. "
            "Use --device INDEX to specify one explicitly."
        )
    name = sd.query_devices(default_input_idx)["name"]
    logger.info("Using default input device: %s (index %d)", name, default_input_idx)
    return default_input_idx, name


def prompt_device_selection() -> int:
    """
    Interactively prompt the user to select an audio input device.

    Returns the chosen device index.
    """
    device_index, device_name = find_blackhole_device()

    if device_index is not None:
        print(f"Detected BlackHole device:  {device_index}: {device_name}\n")
        choice = input("Use this device? (Y/n): ").strip().lower()
        if choice in ("", "y", "yes"):
            logger.info("Using BlackHole device: %s (index %d)", device_name, device_index)
            return device_index

    print("BlackHole device not detected (or not selected).\n")
    print_input_devices()
    selected = int(input("Enter device index: ").strip())
    logger.info("User selected device index: %d", selected)
    return selected
