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
import pandas as pd

from ics.cobraCharmer import cobraState
from ics.fpsActor.utils import blindMoveCalib
from ics.fpsActor.utils import dotGeometry
from ics.fpsActor.utils import dotState

SCAN_STEP_FRACTION = 0.05
"""How much deeper each flat of the flux scan drives the fleet."""

HIDDEN_FLUX = 0.01
"""Residual flux at or below which a cobra counts as hidden and is left alone.

A cobra above it is short of its dot rather than past it: the approach is always from the
entry side, so more depth is the only correction needed and the never-reverse rule holds.

The level is a compromise the scans measure directly.  Replaying the three 13-point scans
of 2026-07-24 and 2026-09-01 against this rule, a cobra stops about three times short of
the depth it went on to reach in the same scan; at 0.05 it stopped ten to fifteen times
short.  Going lower costs more than it buys: a cobra whose floor lies above the level
never satisfies it, is stepped on every remaining flat and climbs the far side of its own
curve, and at 0.005 that happens to some hundred cobras against forty here.

What the level cannot do is beat its own units.  flux_ratio_norm divides by the parked
monitor fibres, but their reading moved 0.858 to 0.972 between two scans on one night and
jumped 15 per cent within a scan, so any absolute figure is applied to a scale that
wanders.  Stopping instead when a flat fails to improve on the last reaches each cobra's
own floor in all three scans and needs no level at all -- see the note on observe().
"""

PLAUSIBLE_FRACTION = (-1.0, 2.0)
"""How far outside its dot an estimate may put a cobra before it is disbelieved.

Generous, because a cobra that never entered the dot is a legitimate outcome.  What it
excludes is an estimate that has diverged -- a wrong-side hard stop, a stalled axis, a
run of moves that the map cannot account for.  Acting on one of those commands a large
open-loop push in an unknown direction, which is how a cobra ends up out of its dot and
fully lit for the rest of the sequence.
"""

DOT_PHI_SCALING_MODES = ('resetBeforeBlind', 'holdThroughRamp')
"""How the adaptive phi on-time scale is used on a dot cobra's convergence.

The two blind moves, the ramp landing and the push, always run at scale 1.0: the scale
in the direction they use is learned from small ramp steps and fits measurement noise,
and on a move nobody measures it can only add error.  'resetBeforeBlind' keeps the
learned scale on every closed-loop move before them; 'holdThroughRamp' runs every phi
move of the dot cobra at 1.0.  Scales start each convergence at 1.0, so in effect
'holdThroughRamp' keeps the dot cobra from learning one.
"""

DEFAULT_DOT_PHI_SCALING = 'resetBeforeBlind'


def dotDepths(convergenceCfg, dotLanding=None, dotTarget=None, noBlindMove=False):
    """Depths a dot convergence lands its cobras at and pushes them to, in dot fraction.

    Parameters
    ----------
    convergenceCfg : `dict`
        The fps `convergence` config section; its `dotLanding` and `dotTarget` apply where
        the keyword is None, and `dotGeometry.RAMP_LANDING_FRACTION` and
        `BLIND_TARGET_FRACTION` where the section has none.
    dotLanding, dotTarget : `float`, optional
        moveToPfsDesign keywords.
    noBlindMove : `bool`
        The cobras stay at the landing, so there is no target.

    Returns
    -------
    (`float`, `float` or None)
        The landing and the target, None if `noBlindMove`.

    Raises
    ------
    `ValueError`
        Unless 0 < landing < target < 1, the push going deeper into the dot; with
        `noBlindMove`, unless 0 < landing < 1.
    """
    landing = float(convergenceCfg.get('dotLanding', dotGeometry.RAMP_LANDING_FRACTION)
                    if dotLanding is None else dotLanding)
    if noBlindMove:
        if not 0 < landing < 1:
            raise ValueError(f'dot depths need 0 < dotLanding < 1, not dotLanding={landing}')
        return landing, None

    target = float(convergenceCfg.get('dotTarget', dotGeometry.BLIND_TARGET_FRACTION)
                   if dotTarget is None else dotTarget)
    if not 0 < landing < target < 1:
        raise ValueError(f'dot depths need 0 < dotLanding < dotTarget < 1, not '
                         f'dotLanding={landing}, dotTarget={target}')
    return landing, target


def unscaledPhiRows(toDot, nIter, mode=DEFAULT_DOT_PHI_SCALING):
    """Which phi moves of a convergence run at scale 1.0, per iteration and cobra.

    Parameters
    ----------
    toDot : `numpy.ndarray` of `bool`
        Per commanded cobra, True for a cobra sent to a black dot.
    nIter : `int`
        Iterations of the convergence; its last is the ramp landing.
    mode : `str`
        One of `DOT_PHI_SCALING_MODES`.

    Returns
    -------
    `numpy.ndarray` of `bool`, (nIter, len(toDot))
        The `unscaledPhi` argument of `engineer.moveThetaPhi`.  Rows of cobras not sent
        to a dot are all False.
    """
    if mode not in DOT_PHI_SCALING_MODES:
        raise ValueError(f'dotPhiScaling must be one of {DOT_PHI_SCALING_MODES}, not {mode!r}')

    toDot = np.asarray(toDot, dtype=bool)
    rows = np.zeros((nIter, len(toDot)), dtype=bool)
    if mode == 'holdThroughRamp':
        rows[:, toDot] = True
    else:
        rows[-1, toDot] = True
    return rows


def withoutInvalidTurnedDots(toBlackDot, isInvalid, dotThetaOffset):
    """Drop the cobras whose target fails validation from the dots, when the dots are turned.

    An invalid target is normally parked on its black dot, a vetted position.  A turned dot
    (`dotGeometry.dotPositions`) is not vetted: it is the target itself, so parking there
    is the move that failed validation.

    Parameters
    ----------
    toBlackDot, isInvalid : `numpy.ndarray` of `bool`, (nCobras,)
    dotThetaOffset : `float`
        Degrees the dots are turned; 0 leaves `toBlackDot` unchanged.

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`) of `bool`, (nCobras,)
        The dot cobras kept, and the ones left out.
    """
    unsafe = toBlackDot & isInvalid if dotThetaOffset else np.zeros_like(toBlackDot)
    return toBlackDot & ~unsafe, unsafe


def loadRunInputs(cc, dataDir=None):
    """Move history and cobra bookkeeping from the run directory just written.

    Parameters
    ----------
    cc : `CobraCoach`
    dataDir : `str`, optional
        Run data directory; defaults to the current one of `cc`.

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
    dataDir = pathlib.Path(cc.runManager.dataDir if dataDir is None else dataDir)
    moves = np.load(dataDir / 'moves.npy')
    filtering = np.load(dataDir / 'cobra_filtering.npz')

    rows = moves[0] if moves.ndim == 3 else moves
    rowCobraId = np.asarray(filtering['final_moving_cobras'], dtype=int)
    dotGlobalIdx = np.flatnonzero(filtering['to_black_dot'])

    if len(rows) != len(rowCobraId):
        raise ValueError(f'moves has {len(rows)} rows but {len(rowCobraId)} cobra ids')

    return rows, rowCobraId, dotGlobalIdx


def makeTracker(cc, cmd=None, blindIterations=0):
    """Seed a position estimate for every cobra from the convergence just run.

    Parameters
    ----------
    cc : `CobraCoach`
    cmd : optional
    blindIterations : `int`
        Final iterations read as undetected, whatever the run recorded.  Behind a real dot
        the last ramp step is not measured, so the push departs from a prediction; on dots
        turned into view (`dotGeometry.dotPositions`) it is measured, and discarding it
        here gives the push the same starting point without altering the run directory.

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

    if blindIterations:
        rows = rows.copy()
        rows['detected'][:, -blindIterations:] = False
        if cmd is not None:
            cmd.inform(f'text="blindMove: the last {blindIterations} iteration(s) read as '
                       f'undetected, so the estimate is predicted rather than measured"')

    nCobras = len(cc.allCobras)
    armLength, _ = dotGeometry._sgfmArrays()
    gains = np.full(nCobras, dotState.DEFAULT_GAIN)
    tracker = dotState.DotTracker.fromMoves(cc.calibModel, rows, rowCobraId, gains,
                                            armLength, nCobras)
    return tracker, dotGlobalIdx


def phiBeforeLanding(cc):
    """Measured phi, radians, at the start of the convergence's last move, per cobra.

    The last move of a dot cobra is the ramp landing, so this is the angle it set off
    from.  NaN for a cobra the run did not move or did not see at that frame.
    """
    rows, rowCobraId, _ = loadRunInputs(cc)
    phi = np.full(len(cc.allCobras), np.nan)
    if rows.shape[1] < 2:
        return phi
    seen = rows['detected'][:, -2]
    phi[rowCobraId[seen]] = rows['phiAngle'][seen, -2]
    return phi


def aimThroughCalibration(cc, wanted, gains, landingFraction, phiCenter, direction, halfDot):
    """Depth to command so that each cobra arrives at the depth `wanted`.

    See `blindMoveCalib.aimedCommand`.  The start of the landing is the measured
    `phiBeforeLanding`, so a cobra not seen there keeps `wanted`; a cobra the calibration
    does not calibrate is aimed with gain 1.0, which also keeps it.

    Parameters
    ----------
    cc : `CobraCoach`
    wanted : `numpy.ndarray` of `float`, (nCobras,)
        Dot fraction.
    gains : `pandas.DataFrame`
        `cobra_blind_move_gain` rows, indexed by cobra_id.
    landingFraction : `float`
    phiCenter, direction, halfDot : `numpy.ndarray`, (nCobras,)
        As `dotGeometry.computeDotAngles`.

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`)
        Depth to command, and the mask of cobras aimed through calibrated gains.
    """
    landingGain, targetGain, calibrated = blindMoveCalib.gainsFor(
        gains, np.arange(1, len(wanted) + 1))

    phiLandingTarget = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction,
                                                        landingFraction)
    command, aimed = blindMoveCalib.aimedCommand(wanted, landingFraction, phiBeforeLanding(cc),
                                                 phiLandingTarget, 2 * direction * halfDot,
                                                 landingGain, targetGain)
    return command, aimed & calibrated


def blindMoveToDots(cc, tracker, dotGlobalIdx, cmd=None, targetFraction=None,
                    deltaFraction=None, stillLit=None, requireHidden=True,
                    dotThetaOffset=0.0, gains=None, landingFraction=dotGeometry.RAMP_LANDING_FRACTION):
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
        Depth every cobra is pushed to; `dotGeometry.BLIND_TARGET_FRACTION`, the dot
        centre, if None.
    deltaFraction : `float`, optional
        Step each cobra this much deeper than the tracker says it already is, instead of
        aiming at a depth.  The flux scan sweeps this way, which keeps each cobra's depth
        known individually rather than assuming they share one.
    stillLit : `numpy.ndarray` of `bool`, optional
        Per-cobra, over all cobras.  When given, only these are moved.  A sequence whose
        job is to hide rather than to map the curve leaves a cobra alone once its flux
        says it is behind the dot, so the ones already there stop accumulating open-loop
        error while the rest catch up.
    requireHidden : `bool`
        Push only the dot cobras the camera has lost.  A cobra still in view is normally
        one whose ramp fell short, and sizing an open-loop move from an estimate already
        proved wrong is worse than leaving it.  On dots turned into view nobody is lost,
        so the measurement of the push needs this off.
    dotThetaOffset : `float`
        Degrees the dots are turned about their cobra centres, see `dotGeometry.dotPositions`.
    gains : `pandas.DataFrame`, optional
        Blind-move calibration to aim through, see `aimThroughCalibration`.  Ignored with
        `deltaFraction`.
    landingFraction : `float`
        Depth the ramp landed at.

    Returns
    -------
    `int`
        Number of cobras commanded.
    """
    detected = cc.cobraInfo['detected']
    hidden = dotGlobalIdx[~detected[dotGlobalIdx]] if requireHidden else dotGlobalIdx
    if stillLit is not None:
        hidden = hidden[stillLit[hidden]]
    if cmd is not None:
        which = ('still lit' if stillLit is not None
                 else 'hidden' if requireHidden else 'sent to a dot, seen or not')
        cmd.inform(f'text="blindMove: {len(hidden)}/{len(dotGlobalIdx)} dot cobras {which}"')
    if not len(hidden):
        return 0

    nCobras = len(cc.allCobras)
    _, phiCenter, _, _, _, direction, halfDot = dotGeometry.computeDotAngles(
        cc, dotThetaOffset=dotThetaOffset)

    if deltaFraction is not None:
        tracker.lastPush = None
        with np.errstate(invalid='ignore'):
            atFraction = 0.5 + (tracker.phi - phiCenter) / (2 * direction * halfDot)
        fractions = atFraction + deltaFraction
    else:
        depth = dotGeometry.BLIND_TARGET_FRACTION if targetFraction is None else targetFraction
        fractions = np.full(nCobras, float(depth))
    command = fractions
    if gains is not None and deltaFraction is None:
        # Before anything moves: a calibration that cannot be applied leaves the push
        # unaimed rather than the convergence failed.
        try:
            command, aimed = aimThroughCalibration(cc, fractions, gains, landingFraction,
                                                   phiCenter, direction, halfDot)
        except Exception as e:
            command = fractions
            if cmd is not None:
                reason = str(e).replace('"', "'")
                cmd.warn(f'text="blindMove: calibration not applied ({reason}); unaimed"')
        else:
            if cmd is not None:
                cmd.inform(f'text="blindMove: {int(aimed[hidden].sum())}/{len(hidden)} cobras '
                           f'aimed through the calibration, the others at their wanted depth"')
    phiTarget = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction, command)

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
    movable = np.asarray(movable, dtype=int)
    if nCommanded:
        # cc.pfi.moveSteps is pure FPGA: a genuine open-loop move, and no per-call MCS
        # frame.  phiFast=False because the steps are sized through the slow map, which
        # every in-patrol move of the ramp ran on; sending slow-sized steps through the
        # fast map overshoots badly.  The push is blind, so it runs at phi scale 1.0
        # (see DOT_PHI_SCALING_MODES).
        with cobraState.unscaled(movable, motor='phi'):
            cc.pfi.moveSteps(cobras, np.zeros(nCommanded, dtype=int),
                             np.asarray(phiSteps, dtype=int), phiFast=False)
        # Only now is the cobra where the estimate should say it is.
        tracker.applied(movable, phiSteps)
    tracker.lastPush = pd.DataFrame(dict(cobraIdx=movable, dotTarget=fractions[movable],
                                         dotCommand=command[movable],
                                         phiPushTarget=phiTarget[movable]))

    if cmd is not None:
        cmd.inform(f'text="blindMove: commanded {nCommanded}/{len(hidden)} '
                   f'({nNoEstimate} with no estimate, '
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


BLIND_MOVE_TABLE = 'cobra_blind_move'
"""opdb table the rows of `blindMoveRecords` go to."""


def rampResidual(db, visit, iteration, nCobras):
    """Distance, mm, of each cobra from its target at one iteration of a visit.

    Parameters
    ----------
    db : `pfs.utils.database.opdb.OpDB`
    visit, iteration, nCobras : `int`

    Returns
    -------
    `numpy.ndarray` of `float`, (nCobras,)
        Indexed by cobra index, NaN where the cobra was not matched.
    """
    rows = db.query_dataframe(
        'SELECT t.cobra_id, t.pfi_target_x_mm tx, t.pfi_target_y_mm ty, '
        'm.pfi_center_x_mm mx, m.pfi_center_y_mm my, m.spot_id '
        'FROM cobra_target t JOIN cobra_match m USING (pfs_visit_id, iteration, cobra_id) '
        f'WHERE t.pfs_visit_id = {int(visit)} AND t.iteration = {int(iteration)}')
    out = np.full(nCobras, np.nan)
    seen = rows.spot_id.to_numpy() != -1          # an unmatched row keeps a stale position
    out[rows.cobra_id.to_numpy()[seen] - 1] = np.abs(
        (rows.mx.to_numpy()[seen] - rows.tx.to_numpy()[seen])
        + 1j * (rows.my.to_numpy()[seen] - rows.ty.to_numpy()[seen]))
    return out


def blindMoveRecords(cc, tracker, dotGlobalIdx, visit, db, positions, landingFraction, *,
                     goHome, dataDir, theta):
    """The landing and the push of every dot cobra, as rows of `BLIND_MOVE_TABLE`.

    `positions` is the frame after the push; the run directory holds the angles before and
    after the landing, the convergence's last move, and tracker.lastPush what the push
    was commanded to.

    Parameters
    ----------
    cc : `CobraCoach`
    tracker : `dotState.DotTracker`
    dotGlobalIdx : `numpy.ndarray` of `int`
        Cobras sent to a dot.
    visit : `int`
    db : `pfs.utils.database.opdb.OpDB`
        Source of the distance from target before the landing.
    positions : `numpy.ndarray` of `complex`, (nCobras,)
        PFI mm after the push, by cobra index, as `CobraCoach.exposeAndExtractPositions`
        returns them; read with cc.cobraInfo['detected'] of that frame.
    landingFraction : `float`
        Depth the ramp landed at.
    goHome : `bool`
        Whether the convergence started with a move home, which takes cobra_target
        iteration 0.
    dataDir : `str`
        Run data directory of the convergence.
    theta : `numpy.ndarray` of `float`, (nCobras,)
        Local theta, radians, the dot cobras were sent to; picks the IK branch.

    Returns
    -------
    `pandas.DataFrame`
        One row per cobra of `dotGlobalIdx`.  Angles are radians, NaN where the cobra was
        not seen; push columns are NaN for a cobra not pushed.
    """
    rows, rowCobraId, _ = loadRunInputs(cc, dataDir=dataDir)
    if rows.shape[1] < 2:
        raise ValueError(f'{rows.shape[1]} moves: no landing to measure')
    nCobras = len(cc.allCobras)

    def perCobra(values):
        out = np.full(nCobras, np.nan)
        out[rowCobraId] = values
        return out

    phiStart = perCobra(np.where(rows['detected'][:, -2], rows['phiAngle'][:, -2], np.nan))
    phiLanded = perCobra(np.where(rows['detected'][:, -1], rows['phiAngle'][:, -1], np.nan))

    idx = np.asarray(dotGlobalIdx, dtype=int)
    pushed = np.where(cc.cobraInfo['detected'][idx], np.asarray(positions)[idx], np.nan)
    phiPushed = dotGeometry.phiNearTheta(cc, idx, pushed, np.asarray(theta)[idx])

    # Phi at a depth is the same on turned dots as on real ones.
    _, phiCenter, _, _, _, direction, halfDot = dotGeometry.computeDotAngles(cc)
    phiLandingTarget = dotGeometry.computePhiAtFraction(phiCenter, halfDot, direction,
                                                        landingFraction)
    columns = ['dotTarget', 'dotCommand', 'phiPushTarget']
    push = (pd.DataFrame(columns=['cobraIdx'] + columns) if tracker.lastPush is None
            else tracker.lastPush).set_index('cobraIdx').reindex(idx)

    # The landing starts after move nIter - 2, which is cobra_target iteration nIter - 1
    # when a move home took iteration 0.
    nIter = rows.shape[1]
    residual = rampResidual(db, visit, nIter - 1 if goHome else nIter - 2, nCobras)

    return pd.DataFrame(dict(
        pfs_visit_id=int(visit), cobra_id=idx + 1, dot_landing=float(landingFraction),
        dot_target=push.dotTarget.to_numpy(dtype=float),
        dot_command=push.dotCommand.to_numpy(dtype=float),
        phi_landing_target=phiLandingTarget[idx],
        phi_push_target=push.phiPushTarget.to_numpy(dtype=float),
        phi_start=phiStart[idx], phi_landed=phiLanded[idx], phi_pushed=phiPushed,
        ramp_residual=residual[idx]))
