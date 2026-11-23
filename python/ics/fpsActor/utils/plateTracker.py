"""The cobra plate fps hands cobraCoach for every MCS frame, and its record in opdb.

The cobra plate moves against the fiducial frame the MCS-to-PFI transform is fitted on.  The
anchor cobras, whose fibre stays put on the plate, give where it is: `plateFrame.fitPlate`
carries their FIBER_REFERENCE positions onto where a frame measured them.  cobraCoach puts the
cobra centres where the plate returned here carries the XML centres.
"""
import logging
import warnings

import numpy as np
import pandas as pd

from ics.cobraCharmer import plateFrame

TABLE = 'cobra_plate_transformation'
PLATE_FRAMES = 3
"""Fewest frames of a visit the plate is fitted on: a median that can reject one bad frame."""
MAX_FAILURES = 5
"""Inserts failing in a row after which recording stops."""


class PlateTracker:
    """`CobraCoach.plateSource` for fps.

    For every frame: the anchors' fit on that frame alone, and the plate applied, the identity
    until `plateFrames` frames of the visit are in, then the fit on the anchors' median
    positions over all the visit's frames so far.  When that fit cannot be made, the last
    plate applied in the visit is kept.  With `enabled` False the identity is always applied.
    Both are written to opdb `TABLE`, one row per frame.  A new visit (frame id // 100)
    starts a new stack.

    Parameters
    ----------
    cc : `CobraCoach`
        Gives the anchors' reference positions, ``cc.calibModel.fiberReference``.
    enabled : `bool`
        Whether the plate is applied.
    plateFrames : `int`
        Frames the plate needs; raised to `PLATE_FRAMES` if lower.
    connectToDB : callable, optional
        Returns the opdb connection the rows are inserted with; None records nothing.
        Recording stops after `MAX_FAILURES` failed inserts in a row.
    logger : `logging.Logger`, optional
    """

    def __init__(self, cc, enabled, plateFrames=PLATE_FRAMES, connectToDB=None, logger=None):
        self.cc = cc
        self.enabled = bool(enabled)
        self.plateFrames = max(PLATE_FRAMES, int(plateFrames))
        self.connectToDB = connectToDB
        self.failures = 0
        self.logger = logger if logger is not None else logging.getLogger('plateTracker')
        self.visit = None
        self.stack = []
        self.applied, self.appliedFrames = plateFrame.IDENTITY, 0

    def __call__(self, frameNum, measured):
        """The plate to apply for one frame.

        Parameters
        ----------
        frameNum : `int`
            MCS frame id.
        measured : `numpy.ndarray` of `complex`, (nCobras,)
            Positions of every cobra in the frame, mm; NaN where not detected.

        Returns
        -------
        `plateFrame.PlateFit`
        """
        reference = self.cc.calibModel.fiberReference
        anchors = np.flatnonzero(np.isfinite(reference))
        visit = int(frameNum) // 100
        if visit != self.visit or any(len(s) != len(anchors) for s in self.stack[:1]):
            self.visit, self.stack = visit, []
            self.applied, self.appliedFrames = plateFrame.IDENTITY, 0
        positions = np.asarray(measured, dtype=complex)[anchors]
        self.stack.append(positions)
        frameFit = plateFrame.fitPlate(positions, reference[anchors])

        if self.enabled and len(self.stack) >= self.plateFrames:
            stack = np.array(self.stack)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)              # an anchor never seen
                median = np.nanmedian(stack.real, axis=0) + 1j * np.nanmedian(stack.imag, axis=0)
            fit = plateFrame.fitPlate(median, reference[anchors])
            if fit is not None:
                self.applied, self.appliedFrames = fit, len(self.stack)
        self.logger.info(f'frame {frameNum}: own fit '
                         f'{frameFit.describe() if frameFit is not None else "none"}; applied '
                         f'{self.applied.describe()} over {self.appliedFrames} frames')
        self._record(frameNum, frameFit)
        return self.applied

    def _record(self, frameNum, frameFit):
        """Insert the row of one frame; a failure is logged, and recording stops after
        `MAX_FAILURES` in a row."""
        if self.connectToDB is None:
            return
        try:
            self.connectToDB().insert_dataframe(TABLE, pd.DataFrame([row(frameNum, frameFit, self.applied,
                                                                         self.appliedFrames)]))
            self.failures = 0
        except Exception as e:
            self.failures += 1
            stop = self.failures >= MAX_FAILURES
            self.logger.warning(f'frame {frameNum}: plate not recorded in {TABLE} ({e})'
                                + ('; recording stopped' if stop else ''))
            if stop:
                self.connectToDB = None


def row(frameNum, frameFit, applied, appliedFrames):
    """One `TABLE` row: the frame's own fit (None when it could not be made) and the plate applied."""

    def columns(fit, prefix):
        if fit is None:
            return {f'{prefix}x0': None, f'{prefix}y0': None, f'{prefix}scale': None, f'{prefix}theta': None,
                    f'{prefix}n_anchors': None, f'{prefix}rms': None}
        return {f'{prefix}x0': fit.a.real, f'{prefix}y0': fit.a.imag, f'{prefix}scale': abs(fit.b),
                f'{prefix}theta': float(np.angle(fit.b)), f'{prefix}n_anchors': fit.nUsed,
                f'{prefix}rms': None if np.isnan(fit.rmsUm) else fit.rmsUm}

    return dict(mcs_frame_id=int(frameNum), **columns(frameFit, 'frame_'), **columns(applied, ''),
                n_frames=int(appliedFrames))
