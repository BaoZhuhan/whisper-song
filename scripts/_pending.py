"""Shared phase-1 guard for commands not implemented yet."""
import sys

def pending(name: str) -> None:
    raise SystemExit(f"{name} is scaffolded but not implemented in phase 1")

