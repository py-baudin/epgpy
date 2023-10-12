""" Implements the Extended Phase Graph NMR signal model

    core.epglib:
        Main EPG bricks and functions
    core.iopulse:
        I/O for pulse files

"""
import pathlib
from .version import __version__

from .core import *
from . import core as epg

PULSEDIR = pathlib.Path(__file__).parent.parent / "resources" / "pulses"
