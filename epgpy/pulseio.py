"""io functions for pulse files"""

import pathlib
import csv
import re
import numpy as np
from . import rfpulse


def load_pulse(filename, duration, resample=None, **kwargs):
    """load pulse file as RFPulse operator"""
    _, values = read_pulse(filename, resample=resample)
    return rfpulse.RFPulse(values, duration, **kwargs)


def read_pulse(filename, resample=None):
    """load pulse file"""
    path = pathlib.Path(filename)
    if path.suffix == ".pta":
        header, values = load_pta(filename)
    else:
        raise NotImplementedError(f"Unknown pulse extension: {path.suffix}")

    if resample and resample < len(values):
        return header, resample_pulse(values, resample)
    return header, values


# list of keys in pta pulse file
PTA_PULSE_KEYS = [
    "PULSENAME",
    "COMMENT",
    "REFGRAD",
    "MINSLICE",
    "MAXSLICE",
    "AMPINT",
    "POWERINT",
    "ABSINT",
]


def load_pta(filename):
    """Load .pta pulse file

    Returns:
    ===
    header: dict
        Values from the pulse header (key names in PTA_PULSE_KEYS)
    values: ndarray
       Complexe values

    """
    header = {}
    index = []
    values = []

    with open(filename, "r") as infile:
        for items in csv.reader(infile, delimiter="\t"):
            if not items or all(not element for element in items):
                # no data on line
                continue

            elif items[0][:-1] in PTA_PULSE_KEYS:
                # if found header key
                header[items[0][:-1]] = items[1]

            elif len(items) == 3 and items[2][0] == ";":
                # if found value
                index.append(int(re.sub("[; ()]", "", items[2])))
                values.append(float(items[0]) * np.exp(1j * float(items[1])))

            else:
                raise IOError('Could not parse line: "%s"' % items)

    values = np.r_[values][np.argsort(index)]
    return header, values


def resample_pulse(values, nsample):
    """resample pulse (complex array) using given sample size"""
    n = len(values)
    xspace = np.linspace(0, n - 1, nsample)
    return np.interp(xspace, np.arange(len(values)), values)
