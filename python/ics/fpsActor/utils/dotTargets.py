"""Per-cobra black-dot depths, read from the calibration product.

The depth at which a cobra hides best is a property of that cobra, not of the fleet: the
measured optima span 0.21 to 0.80 in dot fraction with a real cobra-to-cobra spread of
158 micrometres, so a single global value gives up most of what is available.  This reads
the fitted optima so the blind move can send each cobra to its own.

The file is the `cobraDotTarget` product of pfs_instdata, resolved through the butler
like the black dots themselves, so a new calibration is a data release rather than a
code release.

Kept out of dotGeometry, which is stateless and hardware-free, and out of FpsCmd, which
should stay glue.

A missing or unreadable file is not fatal: every cobra falls back to the global default
and the convergence proceeds.  A calibration is an improvement, never a prerequisite.
"""
import os

import numpy as np
import pandas as pd
from pfs.utils.butler import Butler

PRODUCT = 'cobraDotTarget'
"""Butler key of the calibration file, `pfi/dot/cobra_dot_target.csv` in pfs_instdata."""

TARGET_BOUNDS = (0.15, 0.85)
"""Fractions outside this are rejected rather than clamped.

The scan that produced the file covered roughly 0.26 to 0.86, so a fitted optimum outside
that is an extrapolation of the profile rather than a measurement of it.  Clamping such a
value to the boundary would still assert a depth nobody observed.
"""


def loadDotTargets(nCobras, path=None, cmd=None):
    """Fitted dot depth per cobra, in dot fraction.

    Parameters
    ----------
    nCobras : `int`
        Length of the returned array; cobra ids outside 1..nCobras are ignored.
    path : `str`, optional
        Calibration file; defaults to the `cobraDotTarget` product in pfs_instdata.
    cmd : optional
        Command handle for inform/warn messages.

    Returns
    -------
    `numpy.ndarray` of `float`, (nCobras,)
        Dot fraction, NaN where the cobra has no usable measurement.  All-NaN if the file
        is missing or unreadable, which reproduces the uncalibrated behaviour.
    """
    targets = np.full(nCobras, np.nan)

    try:
        if path is None:
            # KeyError here means the pfs_utils in use predates the product.
            path = Butler().getPath(PRODUCT)
        table = pd.read_csv(path, comment='#')

        cobraId = pd.to_numeric(table.cobraId, errors='coerce').to_numpy()
        fraction = pd.to_numeric(table.dotTargetFraction, errors='coerce').to_numpy()

        # An unset flag means "not calibrated": the file carries a filler value in that
        # column so every cobra has a row, and returning it would pass the fleet default
        # off as a measurement.
        calibrated = table.calibrated.astype(str).str.strip().str.lower()
        calibrated = calibrated.isin(('true', '1', 'yes')).to_numpy()

        keep = (calibrated & np.isfinite(cobraId) & np.isfinite(fraction)
                & (cobraId >= 1) & (cobraId <= nCobras)
                & (fraction >= TARGET_BOUNDS[0]) & (fraction <= TARGET_BOUNDS[1]))

        targets[cobraId[keep].astype(int) - 1] = fraction[keep]

        if cmd is not None:
            when = ''
            try:
                when = f', {pd.Timestamp(os.path.getmtime(path), unit="s"):%Y-%m-%d}'
            except OSError:
                pass
            found = int(np.isfinite(targets).sum())
            median = np.nanmedian(targets) if found else np.nan
            cmd.inform(f'text="dotCalib: {os.path.basename(path)}{when} — '
                       f'{found} of {nCobras} calibrated, median {median:.3f}"')

    except Exception as e:
        if cmd is not None:
            cmd.warn(f'text="dotCalib: read failed ({e}); using the global default"')

    return targets


def resolveTargets(nCobras, default, path=None, cmd=None):
    """Dot fraction for every cobra, measured where possible and `default` elsewhere.

    Parameters
    ----------
    nCobras : `int`
    default : `float`
        Fraction for cobras with no measurement.
    path : `str`, optional
    cmd : optional

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`)
        Fraction per cobra, and a mask of which came from the file.
    """
    targets = loadDotTargets(nCobras, path=path, cmd=cmd)
    measured = np.isfinite(targets)
    return np.where(measured, targets, default), measured


RESPONSE_FILENAME = 'cobra_dot_response.csv'
"""The response file sits beside the dot-target product in pfs_instdata.

Resolved by name rather than as a butler product of its own, so that a new calibration
needs no change in pfs_utils.
"""

GAIN_BOUNDS = (0.5, 2.0)
"""Gains outside this are rejected: a blind move that missed by more than a factor of
two measured something other than the motor."""


def loadResponse(nCobras, path=None, cmd=None):
    """Open-loop gain of each of the two blind moves, per cobra.

    The ramp's landing move and the move to the dot target have their own gains: the
    first is commanded from a measured angle, the second from the tracker's belief, and
    measurement puts their correlation at about 0.4.

    Parameters
    ----------
    nCobras : `int`
    path : `str`, optional
    cmd : optional

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`)
        landingGain and targetGain, each (nCobras,), NaN where there is no measurement.
        Both all-NaN if the file is missing, which reproduces the uncalibrated blind
        move.
    """
    landing = np.full(nCobras, np.nan)
    target = np.full(nCobras, np.nan)

    try:
        if path is None:
            path = os.path.join(os.path.dirname(Butler().getPath(PRODUCT)),
                                RESPONSE_FILENAME)
        table = pd.read_csv(path, comment='#')

        cobraId = pd.to_numeric(table.cobraId, errors='coerce').to_numpy()
        g1 = pd.to_numeric(table.landingGain, errors='coerce').to_numpy()
        g2 = pd.to_numeric(table.targetGain, errors='coerce').to_numpy()
        calibrated = table.calibrated.astype(str).str.strip().str.lower()
        calibrated = calibrated.isin(('true', '1', 'yes')).to_numpy()

        keep = (calibrated & np.isfinite(cobraId) & (cobraId >= 1) & (cobraId <= nCobras)
                & np.isfinite(g1) & np.isfinite(g2)
                & (g1 >= GAIN_BOUNDS[0]) & (g1 <= GAIN_BOUNDS[1])
                & (g2 >= GAIN_BOUNDS[0]) & (g2 <= GAIN_BOUNDS[1]))

        idx = cobraId[keep].astype(int) - 1
        landing[idx] = g1[keep]
        target[idx] = g2[keep]

        if cmd is not None:
            cmd.inform(f'text="dotResponse: {os.path.basename(path)} — '
                       f'{int(keep.sum())} of {nCobras} calibrated, median gains '
                       f'{np.nanmedian(landing):.3f} / {np.nanmedian(target):.3f}"')

    except Exception as e:
        if cmd is not None:
            cmd.warn(f'text="dotResponse: read failed ({e}); blind move uncalibrated"')

    return landing, target
