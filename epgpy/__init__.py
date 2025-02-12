"""Implements the Extended Phase Graph NMR signal model

core.epglib:
    Main EPG bricks and functions
core.iopulse:
    I/O for pulse files

"""

from .core import *
from . import core as epg
from .common import set_array_module, get_array_module
