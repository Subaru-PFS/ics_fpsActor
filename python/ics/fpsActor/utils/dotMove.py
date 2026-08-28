"""Push cobras that have gone behind their black dot deeper into it.

The convergence leaves a dot cobra just inside the rim, where it is no longer measurable:
cobra_match falls back to the dot centre, so the recorded angle is an artefact and the
closed-loop step counts computed from it send the cobra yo-yoing.  The remaining depth
therefore has to be commanded open loop, from an estimate of where the cobra is rather
than from an assumption about it.

This runs after the convergence has written its run directory, and reconstructs
everything it needs from there.  That is what makes it self-contained: the move history,
the cobra ids that go with its rows, and which cobras were sent to a dot are all on disk,
so nothing has to be handed across in memory and the step can be repeated or run on its
own.  It also means the convergence that just finished is simply the most recent run when
the gain is measured, with no special case.
"""
import pathlib

import numpy as np

from ics.fpsActor.utils import dotGeometry
from ics.fpsActor.utils import dotState
from ics.fpsActor.utils import dotTargets

SCAN_STEP_FRACTION = 0.05
"""How much deeper each flat of the flux scan drives the fleet."""

HIDDEN_FLUX = 0.05
"""Residual flux at or below which a cobra counts as hidden and is left alone.

The obscuration optimum is a broad plateau, and interpolating the 2026-07-24 scan curves
at a perfectly placed tip puts 96.8 to 98.4 per cent of cobras under this figure across
fractions 0.35 to 0.50.  A cobra above it is short of its dot rather than past it: the
approach is always from the entry side, so more depth is the only correction needed and
the never-reverse rule holds.
"""

PLAUSIBLE_FRACTION = (-1.0, 2.0)
"""How far outside its dot an estimate may put a cobra before it is disbelieved.

Generous, because a cobra that never entered the dot is a legitimate outcome.  What it
excludes is an estimate that has diverged -- a wrong-side hard stop, a stalled axis, a
run of moves that the map cannot account for.  Acting on one of those commands a large
open-loop push in an unknown direction, which is how a cobra ends up out of its dot and
fully lit for the rest of the sequence.
"""


def loadRunInputs(cc):
    """Move history and cobra bookkeeping from the run directory just written.

    Parameters
    ----------
    cc : `CobraCoach`

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`, `numpy.ndarray`)
        Per-cobra move rows, the global cobra id of each row, and the global indices of
        the cobras sent to a dot.

    Raises
    ------
    FileNotFoundError
        If the convergence did not leave a run directory to read.
    """
    dataDir = pathlib.Path(cc.runManager.dataDir)
    moves = np.load(dataDir / 'moves.npy')
    filtering = np.load(dataDir / 'cobra_filtering.npz')

    rows = moves[0] if moves.ndim == 3 else moves
    rowCobraId = np.asarray(filtering['final_moving_cobras'], dtype=int)
    dotGlobalIdx = np.flatnonzero(filtering['to_black_dot'])

    if len(rows) != len(rowCobraId):
        raise ValueError(f'moves has {len(rows)} rows but {len(rowCobraId)} cobra ids')

    return rows, rowCobraId, dotGlobalIdx


def makeTracker(cc, cmd=None):
    """Seed a position estimate for every cobra from the convergence just run.

    Parameters
    ----------
    cc : `CobraCoach`
    cmd : optional

    Returns
    -------
    (`dotState.DotTracker`, `numpy.ndarray`) or (None, None)
        The tracker and the global indices of the cobras sent to a dot.
    """
    try:
        rows, rowCobraId, dotGlobalIdx = loadRunInputs(cc)
    except Exception as e:
        if cmd is not None:
            cmd.warn(f'text="blindMove: cannot read the run directory ({e}); skipped"')
        return None, None

    nCobras = len(cc.allCobras)
    armLength, _ = dotGeometry._sgfmArrays()
    gains = dotState.loadRecentGains(cc.runManager.rootDir, nCobras, cc.calibModel,
                                     cmd=cmd)
    tracker = dotState.DotTracker.fromMoves(cc.calibModel, rows, rowCobraId, gains,
                                            armLength, nCobras)
    return tracker, dotGlobalIdx


def blindMoveToDots(cc, tracker, dotGlobalIdx, cmd=None, targetFraction=None,
                    targetOffset=0.0, deltaFraction=None, stillLit=None):
    """Step every hidden dot cobra to its target depth, open loop.

    The tracker is advanced by whatever is commanded, so calling this repeatedly -- as
    the flux scan does -- steps the cobra on from where the previous call left it rather
    than from where the convergence ended.

    Parameters
    ----------
    cc : `CobraCoach`
        Source of cc.cobraInfo['detected'], cc.allCobras and cc.pfi.moveSteps.
    tracker : `dotState.DotTracker`
        Position estimate, updated in place.
    dotGlobalIdx : `numpy.ndarray` of `int`
        Cobras sent to a dot.
    cmd : optional
        Command handle for inform/warn messages.
    targetFraction : `float`, optional
        One depth for every cobra, overriding the measured optima.  The flux scan walks
        the fleet through a common series of depths and needs that; a convergence leaves
        it unset so each cobra goes to its own.
    targetOffset : `float`
        Added to every target, measured or default.  A fine-tuning sequence walks the
        whole fleet a little shallower or deeper without discarding the per-cobra
        differences the calibration measured.
    deltaFraction : `float`, optional
        Step each cobra this much deeper than the tracker says it already is, instead of
        aiming at a depth.  The flux scan sweeps this way, which keeps each cobra's depth
        known individually rather than assuming they share one.
    stillLit : `numpy.ndarray` of `bool`, optional
        Per-cobra, over all cobras.  When given, only these are moved.  A sequence whose
        job is to hide rather than to map the curve leaves a cobra alone once its flux
        says it is behind the dot, so the ones already there stop accumulating open-loop
        error while the rest catch up.

    Returns
    -------
    `int`
        Number of cobras commanded.
    """
    detected = cc.cobraInfo['detected']
    hidden = dotGlobalIdx[~detected[dotGlobalIdx]]
    if stillLit is not None:
        hidden = hidden[stillLit[hidden]]
    if cmd is not None:
        cmd.inform(f'text="blindMove: {len(hidden)}/{len(dotGlobalIdx)} dot cobras '
                   f'{"still lit" if stillLit is not None else "hidden"}"')
    if not len(hidden):
        return 0

    nCobras = len(cc.allCobras)
    _, phiCenter, _, _, _, direction, halfDot = dotGeometry.computeDotAngles(cc)

    if deltaFraction is not None:
        with np.errstate(invalid='ignore'):
            atFraction = 0.5 + (tracker.phi - phiCenter) / (2 * direction * halfDot)
        fractions = atFraction + deltaFraction
        measured = np.zeros(nCobras, dtype=bool)
    elif targetFraction is None:
        fractions, measured = dotTargets.resolveTargets(
            nCobras, dotGeometry.BLIND_TARGET_FRACTION, cmd=cmd)
    else:
        fractions = np.full(nCobras, float(targetFraction))
        measured = np.zeros(nCobras, dtype=bool)
    fractions = fractions + targetOffset
    phiTarget = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction, fractions)

    movable, cobras, phiSteps = [], [], []
    nNoEstimate = nImplausible = 0
    for cobraId in hidden:
        phiNow = tracker.phi[cobraId]
        if not np.isfinite(phiNow):
            nNoEstimate += 1
            continue

        # Where the estimate says the cobra is, in its own dot.  An estimate that has
        # diverged is worse than none: it sizes a confident move in the wrong direction.
        atFraction = 0.5 + (phiNow - phiCenter[cobraId]) / (
            2 * direction[cobraId] * halfDot[cobraId])
        if not (PLAUSIBLE_FRACTION[0] <= atFraction <= PLAUSIBLE_FRACTION[1]):
            nImplausible += 1
            continue

        steps = dotState.stepsToTarget(cc.calibModel, cobraId, phiNow,
                                       phiTarget[cobraId], tracker.gain[cobraId],
                                       maxSteps=dotGeometry.MAX_BLIND_STEPS)
        if steps == 0:
            continue
        movable.append(cobraId)
        cobras.append(cc.allCobras[cobraId])
        phiSteps.append(int(steps))

    nCommanded = len(cobras)
    if nCommanded:
        # cc.pfi.moveSteps is pure FPGA: a genuine open-loop move, and no per-call MCS
        # frame.  phiFast=False because the gain is measured against the slow map, which
        # every in-patrol move of the ramp ran on; sending slow-sized steps through the
        # fast map overshoots badly.
        cc.pfi.moveSteps(cobras, np.zeros(nCommanded, dtype=int),
                         np.asarray(phiSteps, dtype=int), phiFast=False)
        # Only now is the cobra where the estimate should say it is.
        tracker.applied(movable, phiSteps)

    if cmd is not None:
        cmd.inform(f'text="blindMove: commanded {nCommanded}/{len(hidden)} '
                   f'({int(measured[hidden].sum())} on a measured target, '
                   f'{nNoEstimate} with no estimate, '
                   f'{nImplausible} with an estimate outside the dot)"')
    return nCommanded


def litFromFlux(fluxDf, nCobras):
    """Which cobras the last flat still saw light from.

    A cobra with no flux row, or an unmeasurable one, is reported not lit.  The
    alternative is to push a cobra nobody measured, which is how one ends up driven
    through its dot and out the far side with nothing watching.

    Parameters
    ----------
    fluxDf : `pandas.DataFrame`
        Columns cobra_id and flux_ratio_norm, for one visit.
    nCobras : `int`

    Returns
    -------
    `numpy.ndarray` of `bool`, (nCobras,) or None
        None when the flat produced no rows at all.  That is drp not writing, not a
        hidden fleet, and the two must not read alike: the second is success and the
        first, taken as success, ends the sequence with every cobra still lit.
    """
    if fluxDf is None or fluxDf.empty:
        return None

    lit = np.zeros(nCobras, dtype=bool)
    cobraId = np.asarray(fluxDf.cobra_id, dtype=int)
    flux = np.asarray(fluxDf.flux_ratio_norm, dtype=float)
    inRange = (cobraId >= 1) & (cobraId <= nCobras)
    with np.errstate(invalid='ignore'):
        above = np.isfinite(flux) & (flux > HIDDEN_FLUX)
    lit[cobraId[inRange & above] - 1] = True
    return lit


DEPTH_FILE = 'dot_depth.csv'


def recordDepths(cc, tracker, dotGlobalIdx, visitId, cmd=None):
    """Write where each dot cobra is, against the visit of the flat just taken.

    dot_roach_flux supplies the flux of that flat; this supplies the depth it was
    measured at, and the two join on (pfs_visit_id, cobra_id).  Nothing else can supply
    it: the cobras no longer share one depth, so it cannot be reconstructed from the scan
    parameters afterwards.

    Called before stepping deeper, so the depth recorded is the one the flat saw.  An
    off-by-one here shifts every curve by one scan step, which is the size of the effect
    being measured.

    Parameters
    ----------
    cc : `CobraCoach`
    tracker : `dotState.DotTracker`
    dotGlobalIdx : `numpy.ndarray` of `int`
    visitId : `int`
        Visit of the flat whose flux this depth belongs to.
    cmd : optional
    """
    _, phiCenter, _, _, _, direction, halfDot = dotGeometry.computeDotAngles(cc)
    path = pathlib.Path(cc.runManager.dataDir) / DEPTH_FILE

    try:
        new = not path.exists()
        with open(path, 'a') as fh:
            if new:
                fh.write('pfs_visit_id,cobra_id,phi,fraction,variance,gain\n')
            for cobraId in np.asarray(dotGlobalIdx, dtype=int):
                phi = tracker.phi[cobraId]
                if not np.isfinite(phi):
                    continue
                fraction = 0.5 + (phi - phiCenter[cobraId]) / (
                    2 * direction[cobraId] * halfDot[cobraId])
                fh.write(f'{int(visitId)},{cobraId + 1},{phi:.6f},{fraction:.5f},'
                         f'{tracker.variance[cobraId]:.3e},{tracker.gain[cobraId]:.4f}\n')
        if cmd is not None:
            cmd.inform(f'text="dotDepth: visit {int(visitId)} recorded to {path.name}"')
    except Exception as e:
        if cmd is not None:
            cmd.warn(f'text="dotDepth: could not record ({e}); the scan is not reducible"')
