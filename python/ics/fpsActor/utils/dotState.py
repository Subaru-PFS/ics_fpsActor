"""Where a cobra is once the camera has stopped seeing it.

The last iterations of a dot convergence are commanded blind: the ramp pushes the tip
across the dot edge, the spot vanishes, and every move after that is open loop.  Until
now the blind move assumed the cobra sat at the fraction the ramp aimed for.  Here the
position is estimated instead, from the moves the cobra was actually given.

Three quantities, at three timescales:

    gain g   how far the cobra moves per map-predicted radian     (this visit, or recent)
    phiNow   where it is now, given the moves since the last look (this iteration)
    steps    what to send to reach a target angle                 (terminal, open loop)

`g` appears twice and must be the same number in both: inside the estimate it propagates
the state, outside it converts a desired angle into steps.  If they disagree the filter
and the command describe different cobras.

Two things the caller must respect.  Rows of `moves` are positions within the commanded
set, not cobra ids -- both indices are required and they are not interchangeable.  And a
hidden cobra still has a `phiAngle`: cobra_match falls back to the dot centre and
positionsToAngles returns phiCenter, so the array carries a plausible angle whether or
not anything was seen.  `detected` is the only gate.
"""
import glob
import os
import time

import numpy as np

RECENT_RUNS = 3
"""How many previous convergences to average a gain over."""

MAX_RUN_AGE_S = 24 * 3600
"""Older runs are ignored.

A run from a previous night was taken against a different calibModel, different on-time
scaling and possibly different motor maps, so its gain describes a machine that no longer
exists -- and blending it in would say nothing about having done so.
"""

PROCESS_NOISE_UM = 18.0
"""Positioning scatter of one move, micrometres of tip travel."""

MEASURE_NOISE_UM = 3.7
"""MCS centroid scatter, micrometres, measured per cobra about its own median."""

MIN_PAIRS = 3
"""Consecutive measured moves below which a gain is not worth fitting."""

MIN_STEPS_FOR_GAIN = 10
"""Moves smaller than this are dominated by quantisation and backlash.

cobraCoach declines to scale on-time below the same figure.
"""

GAIN_BOUNDS = (0.5, 2.0)
"""Gains outside this are not believed: dividing by one would size an unbounded move."""

DEFAULT_GAIN = 1.0

STEP_CEILING = 1.1 / GAIN_BOUNDS[0]
"""Ceiling on the commanded steps, as a multiple of what the map alone asks for.

Derived from the gain bound rather than chosen: the slowest believable cobra needs
1/GAIN_BOUNDS[0] times the map's step count, so anything beyond that is a gain that has
collapsed rather than a cobra that is unusual.  A fleet-wide step clamp cannot do this
job -- 3000 steps is some 160 degrees of phi on the median map.
"""


def mapSpan(calibModel, cobraId, deltaSteps, fromAngle):
    """Angle the motor map predicts after `deltaSteps`, radians.

    The slow map is used throughout -- fast and slow are different calibrations, and a
    gain measured against one cannot size a move through the other.  Direction comes from
    the sign of the move, as the step-to-angle conversion does elsewhere, so the two
    cannot drift apart.

    Parameters
    ----------
    calibModel : `ics.cobraCharmer.pfiDesign.PFIDesign`
    cobraId : `int`
        Zero-based cobra index, global.
    deltaSteps : `float`
        Signed step count, positive opening.
    fromAngle : `float`
        Starting phi, radians.

    Returns
    -------
    `float`
        NaN when the map is unusable or the start lies outside it.
    """
    angles = calibModel.phiOffsets[cobraId]
    steps = (calibModel.posPhiSlowSteps[cobraId] if deltaSteps >= 0
             else calibModel.negPhiSlowSteps[cobraId])

    if len(angles) < 2 or not np.all(np.diff(steps) > 0):
        return np.nan
    if not np.isfinite(fromAngle) or not (angles[0] <= fromAngle <= angles[-1]):
        return np.nan

    cumulative = np.interp(fromAngle, angles, steps)
    return float(np.interp(cumulative + deltaSteps, steps, angles))


def gainFromMoves(rowMoves, calibModel, cobraId):
    """Achieved motion over map-predicted motion, from the measured iterations.

    Signed rather than absolute: the step count carries the direction and the map is
    monotonic, so a cobra moving as commanded gives a positive ratio whichever way it
    turns.  Taking the magnitude would hide a cobra travelling backwards, which is the
    one failure this number exists to expose.

    The last measured move before the cobra vanishes is dropped.  Surviving that move is
    anti-correlated with gain -- the cobras still visible are the ones that moved least --
    so including it biases the estimate low on exactly the cobras that then need it most.

    Parameters
    ----------
    rowMoves : structured `numpy.ndarray`
        One cobra's row of the moves array; needs phiAngle, phiSteps, detected.
    calibModel : `ics.cobraCharmer.pfiDesign.PFIDesign`
    cobraId : `int`
        Zero-based, global.

    Returns
    -------
    `float`
        NaN when fewer than `MIN_PAIRS` usable moves remain.
    """
    phi = np.asarray(rowMoves['phiAngle'], dtype=float)
    steps = np.asarray(rowMoves['phiSteps'], dtype=float)
    detected = np.asarray(rowMoves['detected'], dtype=bool)

    usable = (detected[:-1] & detected[1:]
              & (np.abs(steps[1:]) >= MIN_STEPS_FOR_GAIN))

    # Drop the move into darkness, but only when the cobra actually went dark: if it was
    # seen to the end there is no survival selection and nothing to correct for.
    visible = np.flatnonzero(detected)
    if len(visible) and visible[-1] < len(detected) - 1 and usable.any():
        usable[np.flatnonzero(usable)[-1]] = False

    ratios = []
    for i in np.flatnonzero(usable):
        predicted = mapSpan(calibModel, cobraId, steps[i + 1], phi[i]) - phi[i]
        if np.isfinite(predicted) and predicted != 0:
            ratios.append((phi[i + 1] - phi[i]) / predicted)

    if len(ratios) < MIN_PAIRS:
        return np.nan
    return float(np.median(ratios))


APPLY_MEASURED_GAIN = False
"""Whether the blind move sizes its steps from the gain it measured.

The convergence adapts the motor on-time until a cobra moves as the motor map predicts,
so the measured gain should sit near 1 and applying it as well would take out the same
discrepancy twice.  The measurement is still made -- it is what says whether the on-time
loop is doing its job -- and remains recoverable from moves.npy either way.
"""


def usableGain(gain):
    """The gain if it is believable, else `DEFAULT_GAIN`."""
    if not APPLY_MEASURED_GAIN:
        return DEFAULT_GAIN
    if not np.isfinite(gain) or not (GAIN_BOUNDS[0] <= gain <= GAIN_BOUNDS[1]):
        return DEFAULT_GAIN
    return float(gain)


def estimatePhi(rowMoves, calibModel, cobraId, gain, armLength):
    """Phi at the end of the ramp, and how well it is known.

    Predict through the map and scale by the gain; correct on every iteration the camera
    saw the cobra, and on no other.  Iterations that commanded nothing add no motion and
    no uncertainty.

    Parameters
    ----------
    rowMoves : structured `numpy.ndarray`
        One cobra's row of the moves array.
    calibModel : `ics.cobraCharmer.pfiDesign.PFIDesign`
    cobraId : `int`
        Zero-based, global.
    gain : `float`
    armLength : `float`
        Phi arm length, millimetres, to convert the noise terms into radians.

    Returns
    -------
    (`float`, `float`, `int`)
        Phi in radians, its variance in radians squared, and the number of moves
        propagated without a measurement.  Phi is NaN if the cobra was never seen.
    """
    phi = np.asarray(rowMoves['phiAngle'], dtype=float)
    steps = np.asarray(rowMoves['phiSteps'], dtype=float)
    detected = np.asarray(rowMoves['detected'], dtype=bool)

    visible = np.flatnonzero(detected)
    if not len(visible):
        return np.nan, np.nan, 0

    processVar = (PROCESS_NOISE_UM / 1000.0 / armLength) ** 2
    measureVar = (MEASURE_NOISE_UM / 1000.0 / armLength) ** 2

    estimate = phi[visible[0]]
    variance = measureVar
    blind = 0

    for i in range(visible[0] + 1, len(phi)):
        if steps[i] != 0:
            predicted = mapSpan(calibModel, cobraId, steps[i], estimate)
            if not np.isfinite(predicted):
                return np.nan, np.nan, blind
            estimate += gain * (predicted - estimate)
            variance += processVar
            if not detected[i]:
                blind += 1

        if detected[i]:
            k = variance / (variance + measureVar)
            estimate += k * (phi[i] - estimate)
            variance *= (1 - k)

    return float(estimate), float(variance), blind


def stepsToTarget(calibModel, cobraId, phiNow, phiTarget, gain, maxSteps=None):
    """Steps that take the cobra from `phiNow` to `phiTarget`.

    The map says how many steps the journey is worth; the gain says how much of it the
    cobra actually performs.  So the count is **divided** by the gain: a cobra that
    overshoots the map needs fewer steps, not more.  Multiplying would apply the error
    twice in the same direction.

    Parameters
    ----------
    calibModel : `ics.cobraCharmer.pfiDesign.PFIDesign`
    cobraId : `int`
        Zero-based, global.
    phiNow, phiTarget : `float`
        Radians.
    gain : `float`
    maxSteps : `float`, optional
        Absolute ceiling, applied after the per-cobra one.

    Returns
    -------
    `int`
        Zero when the move cannot be sized.
    """
    if not np.isfinite(phiNow) or not np.isfinite(phiTarget):
        return 0

    angles = calibModel.phiOffsets[cobraId]
    forward = phiTarget >= phiNow
    steps = (calibModel.posPhiSlowSteps[cobraId] if forward
             else calibModel.negPhiSlowSteps[cobraId])
    if len(angles) < 2 or not np.all(np.diff(steps) > 0):
        return 0
    if not (angles[0] <= phiNow <= angles[-1] and angles[0] <= phiTarget <= angles[-1]):
        return 0

    mapSteps = np.interp(phiTarget, angles, steps) - np.interp(phiNow, angles, steps)
    commanded = mapSteps / usableGain(gain)

    # The gain can only ever shorten or lengthen the move by so much; beyond that the
    # gain is wrong rather than the cobra unusual.
    ceiling = STEP_CEILING * abs(mapSteps)
    if maxSteps is not None:
        ceiling = min(ceiling, abs(maxSteps))
    return int(round(np.clip(commanded, -ceiling, ceiling)))


def loadRecentGains(dataRoot, nCobras, calibModel, nRuns=RECENT_RUNS,
                    maxAge=MAX_RUN_AGE_S, cmd=None):
    """Per-cobra gain from the most recent convergences on disk.

    A gain measured over several runs is steadier than one measured over the handful of
    iterations of a single visit, but only while the machine has not changed underneath
    it, hence `maxAge`.

    A run is used only if it carries both the move history and the cobra ids that go with
    it: `moves.npy` records rows in commanded order and nothing else, so without
    `cobra_filtering.npz` the rows cannot be attributed and the file is skipped rather
    than guessed at.  That also excludes motor-map and convergence-test runs, whose
    `moves.npy` has a different shape entirely.

    Parameters
    ----------
    dataRoot : `str`
        Directory holding the dated run directories.
    nCobras : `int`
    calibModel : `ics.cobraCharmer.pfiDesign.PFIDesign`
    nRuns : `int`
    maxAge : `float`
        Seconds.
    cmd : optional
        Command handle for inform/warn messages.

    Returns
    -------
    `numpy.ndarray` of `float`, (nCobras,)
        Median gain per cobra, NaN where no recent run measured it.  The convergence
        that is about to move is itself one of these runs: its moves are on disk by the
        time the blind move runs, so it needs no separate treatment.
    """
    gains = np.full(nCobras, np.nan)
    collected = [[] for _ in range(nCobras)]
    used = []

    try:
        candidates = sorted(glob.glob(os.path.join(dataRoot, '*', 'data', 'moves.npy')),
                            key=os.path.getmtime, reverse=True)
    except OSError as e:
        if cmd is not None:
            cmd.warn(f'text="dotGain: cannot list {dataRoot} ({e}); assuming unit gain"')
        return gains

    now = time.time()
    for path in candidates:
        if len(used) >= nRuns:
            break
        if now - os.path.getmtime(path) > maxAge:
            break

        mapping = os.path.join(os.path.dirname(path), 'cobra_filtering.npz')
        if not os.path.exists(mapping):
            continue
        try:
            moves = np.load(path)
            cobraIds = np.load(mapping)['final_moving_cobras']
        except Exception:
            continue

        rows = moves[0] if moves.ndim == 3 else moves
        if rows.ndim != 2 or len(rows) != len(cobraIds):
            continue

        for row, cobraId in enumerate(np.asarray(cobraIds, dtype=int)):
            if not (0 <= cobraId < nCobras):
                continue
            gain = gainFromMoves(rows[row], calibModel, cobraId)
            if np.isfinite(gain):
                collected[cobraId].append(gain)
        used.append(os.path.basename(os.path.dirname(os.path.dirname(path))))

    for cobraId, values in enumerate(collected):
        if values:
            gains[cobraId] = float(np.median(values))

    if cmd is not None:
        found = int(np.isfinite(gains).sum())
        if used:
            cmd.inform(f'text="dotGain: {len(used)} runs ({", ".join(used)}) — '
                       f'{found} of {nCobras} cobras, median '
                       f'{np.nanmedian(gains) if found else float("nan"):.3f}"')
        else:
            cmd.inform('text="dotGain: no recent run on disk; assuming unit gain"')

    return gains


class DotTracker:
    """Best estimate of where each cobra is, carried across commands.

    Built from the convergence, advanced by every move commanded afterwards, corrected by
    every measurement that arrives.  One blind move at the end of a convergence and ten
    successive ones during a flux scan are the same operation against the same state,
    which is what lets the scan account for the steps it has already sent instead of
    assuming each of them landed.

    Every array is indexed by global cobra id and NaN where the cobra was never seen.
    """

    def __init__(self, calibModel, phi, variance, gain, armLength):
        self.calibModel = calibModel
        self.phi = np.asarray(phi, dtype=float)
        self.variance = np.asarray(variance, dtype=float)
        self.gain = np.asarray(gain, dtype=float)
        self.armLength = np.asarray(armLength, dtype=float)

    @classmethod
    def fromMoves(cls, calibModel, rows, rowCobraId, gains, armLength, nCobras):
        """Seed the estimate by replaying a convergence.

        Parameters
        ----------
        calibModel : `ics.cobraCharmer.pfiDesign.PFIDesign`
        rows : structured `numpy.ndarray`
            Move history, one row per commanded cobra.
        rowCobraId : `numpy.ndarray` of `int`
            Global cobra id of each row.
        gains : `numpy.ndarray` of `float`, (nCobras,)
        armLength : `numpy.ndarray` of `float`, (nCobras,)
        nCobras : `int`

        Returns
        -------
        `DotTracker`
        """
        phi = np.full(nCobras, np.nan)
        variance = np.full(nCobras, np.nan)
        gain = np.array([usableGain(g) for g in gains])

        for row, cobraId in enumerate(np.asarray(rowCobraId, dtype=int)):
            if not (0 <= cobraId < nCobras):
                continue
            phi[cobraId], variance[cobraId], _ = estimatePhi(
                rows[row], calibModel, cobraId, gain[cobraId], armLength[cobraId])

        return cls(calibModel, phi, variance, gain, armLength)

    def known(self, cobraIds):
        """Which of `cobraIds` have a usable estimate."""
        return np.isfinite(self.phi[np.asarray(cobraIds, dtype=int)])

    def stepsTo(self, cobraIds, phiTarget, maxSteps=None):
        """Steps that would take each cobra to its target, zero where unsizeable.

        Parameters
        ----------
        cobraIds : `numpy.ndarray` of `int`
        phiTarget : `numpy.ndarray` of `float`
            Indexed by global cobra id.
        maxSteps : `float`, optional

        Returns
        -------
        `numpy.ndarray` of `int`
        """
        cobraIds = np.asarray(cobraIds, dtype=int)
        return np.array([stepsToTarget(self.calibModel, c, self.phi[c], phiTarget[c],
                                       self.gain[c], maxSteps=maxSteps)
                         for c in cobraIds], dtype=int)

    def applied(self, cobraIds, steps):
        """Advance the estimate by steps that have been commanded.

        Called after the move, not before: the state describes where the cobra is, and it
        is only there once the command has gone out.

        Parameters
        ----------
        cobraIds : `numpy.ndarray` of `int`
        steps : `numpy.ndarray` of `int`
        """
        cobraIds = np.asarray(cobraIds, dtype=int)
        for cobraId, step in zip(cobraIds, np.asarray(steps, dtype=int)):
            if step == 0 or not np.isfinite(self.phi[cobraId]):
                continue
            predicted = mapSpan(self.calibModel, cobraId, step, self.phi[cobraId])
            if not np.isfinite(predicted):
                self.phi[cobraId] = np.nan
                continue
            self.phi[cobraId] += self.gain[cobraId] * (predicted - self.phi[cobraId])
            self.variance[cobraId] += (PROCESS_NOISE_UM / 1000.0
                                       / self.armLength[cobraId]) ** 2

    def observe(self, phiMeasured, detected):
        """Correct the estimate wherever the camera saw the cobra.

        `phiMeasured` carries a plausible angle for every cobra whether or not anything
        was seen -- cobra_match falls back to the dot centre -- so `detected` is the only
        thing that says which entries are measurements.

        Parameters
        ----------
        phiMeasured : `numpy.ndarray` of `float`
            Indexed by global cobra id.
        detected : `numpy.ndarray` of `bool`
        """
        phiMeasured = np.asarray(phiMeasured, dtype=float)
        detected = np.asarray(detected, dtype=bool)

        for cobraId in np.flatnonzero(detected & np.isfinite(phiMeasured)):
            measureVar = (MEASURE_NOISE_UM / 1000.0 / self.armLength[cobraId]) ** 2
            if not np.isfinite(self.phi[cobraId]):
                self.phi[cobraId] = phiMeasured[cobraId]
                self.variance[cobraId] = measureVar
                continue
            k = self.variance[cobraId] / (self.variance[cobraId] + measureVar)
            self.phi[cobraId] += k * (phiMeasured[cobraId] - self.phi[cobraId])
            self.variance[cobraId] *= (1 - k)
