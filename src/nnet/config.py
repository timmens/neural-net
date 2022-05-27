"""This module contains the general configuration of the project."""
from pathlib import Path


SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()
NUM_CLASSES = 10  # digits 0 until 9


__all__ = ["BLD", "SRC"]
