"""dotGeometry.py — pure per-cobra geometry for black-dot hiding.

All functions are stateless and hardware-free.  They take calibration data
directly from a CobraCoach instance (cc) and return numpy arrays indexed by
cobra (length = len(cc.allCobras)).

Convention
----------
phiMin   : phi angle at the near (minimum-phi) dot edge
             = approach entry for CCW (+1), approach entry for CW (-1) is phiMax
phiMax   : phi angle at the far (maximum-phi) dot edge
phiEnter : phi at the approach entry edge  (phiMin for CCW, phiMax for CW)
phiExit  : phi at the approach exit  edge  (phiMax for CCW, phiMin for CW)
direction: +1 CCW (opening arm, increasing phi)
           -1 CW  (closing arm,  decreasing phi)
fraction : 0 = phiEnter, 0.5 = phiCenter, 1 = phiExit
"""

import numpy as np
from ics.fpsActor.utils.alfUtils import sgfm


# Phi cap (deg) per leading "fast" iteration of moveThetaPhi.  Both the phi
# clipping in capCommandedAngle and the matching theta safety offset in
# buildDotRamp use this same schedule.
PHI_CAPS_DEG = (30.0, 45.0)


# ── helpers ──────────────────────────────────────────────────────────────────

def _sgfmArrays():
    """Return (L2, rDot) numpy arrays aligned to cobra index."""
    return sgfm.L2.to_numpy(), sgfm.rDot.to_numpy()


def _applyThetaMargin(thetaTarget, thetaMargin=0.1):
    """Bump theta angles below margin up by 2π (avoid CW hard stop)."""
    t = thetaTarget.copy()
    t[t < thetaMargin] += 2 * np.pi
    return t


# ── public API ────────────────────────────────────────────────────────────────

def computeDotAngles(cc, phiFloor=np.radians(15.0), minRangeDeg=30.0):
    """Compute per-cobra dot geometry for the phi-crossing strategy.

    Parameters
    ----------
    cc : CobraCoach
    phiFloor : float
        Hard-stop margin — arm cannot start below this phi (radians). Default 15°.
    minRangeDeg : float
        Minimum ramp travel required for CCW approach (degrees). Default 30°.
        CCW is chosen only when phiMin >= phiFloor + minRange, guaranteeing
        a full run-up.  Cobras below that threshold use CW.

    Returns
    -------
    thetaStart : ndarray (nCobras,)
        Local theta angle where the phi arm sweeps through the dot center.
    phiCenter : ndarray (nCobras,)
        Phi angle at the dot center.
    phiMin : ndarray (nCobras,)
        Phi angle at the near (minimum-phi) dot edge.
    phiMax : ndarray (nCobras,)
        Phi angle at the far (maximum-phi) dot edge.
    phiEnter : ndarray (nCobras,)
        Phi at the approach entry edge (phiMin for CCW, phiMax for CW).
    direction : ndarray (nCobras,) int
        +1 CCW, -1 CW.
    halfDot : ndarray (nCobras,)
        Half-angle subtended by the dot radius at the elbow (radians).
    """
    L2, rDot = _sgfmArrays()
    nCobras = len(cc.allCobras)

    dotPos = sgfm.xDot.to_numpy() + 1j * sgfm.yDot.to_numpy()

    # IK: theta that places the elbow so phi sweeps through the dot center
    thetaSol, phiSol, _ = cc.pfi.positionsToAngles(cc.allCobras, dotPos)
    thetaStart = _applyThetaMargin(thetaSol[:, 0])
    phiCenter = phiSol[:, 0]

    # Geometry: half-angle of the dot as seen from the elbow
    elbows = cc.pfi.anglesToElbowPositions(cc.allCobras, thetaStart)
    dist = np.abs(elbows - dotPos)
    cosA = np.clip((L2 ** 2 + dist ** 2 - rDot ** 2) / (2 * L2 * dist), -1, 1)
    halfDot = np.arccos(cosA)

    phiMinArr = phiCenter - halfDot  # near edge (smallest phi)
    phiMaxArr = phiCenter + halfDot  # far  edge (largest  phi)

    # Direction choice: CCW only if phiMin gives enough room for a full ramp
    ccwThreshold = phiFloor + np.deg2rad(minRangeDeg)
    direction = np.zeros(nCobras, dtype=int)
    phiEnter = np.zeros(nCobras)

    ccwIdx = np.intersect1d(cc.goodIdx, np.where(phiMinArr >= ccwThreshold)[0])
    cwIdx = np.intersect1d(cc.goodIdx, np.where(phiMinArr < ccwThreshold)[0])

    direction[ccwIdx] = 1
    direction[cwIdx] = -1
    phiEnter[ccwIdx] = phiMinArr[ccwIdx]  # CCW enters from phiMin
    phiEnter[cwIdx] = phiMaxArr[cwIdx]  # CW  enters from phiMax

    return thetaStart, phiCenter, phiMinArr, phiMaxArr, phiEnter, direction, halfDot


def thetaOffsetForPhi(L1, L2, rDot, phi_rad, motorMarginMm=0.05):
    """Per-cobra theta offset magnitude (rad) needed to clear the dot
    footprint when the arm is forced to angle ``phi_rad``.

    The tip-to-base distance at phi is
        D(phi) = sqrt(L1² + L2² − 2·L1·L2·cos(phi))
    so a theta sweep of Δθ moves the tip tangentially around the base
    by D·Δθ.  To guarantee that any tip starting inside the dot
    (worst case: dot-centre) ends up outside the rDot circle, we need
        Δθ ≥ (rDot + motorMarginMm) / D(phi).

    The motor-margin term accounts for the cobra not landing exactly on
    the commanded position (default 50 µm).
    """
    D = np.sqrt(L1**2 + L2**2 - 2*L1*L2*np.cos(phi_rad))
    return (rDot + motorMarginMm) / D


def buildDotRamp(cc, dotCobras, nIter, capIters=2, motorMarginMm=0.05):
    """Build theta/phi starts and phi/theta ramp arrays for dot cobras.

    One-call wrapper used by moveToPfsDesign.  All heavy geometry stays here.

    Parameters
    ----------
    cc : CobraCoach
    dotCobras : array-like of int
        Global cobra indices that should hide behind their black dot.
    nIter : int
        Number of moveThetaPhi iterations (= phi/theta ramp rows).
    capIters : int
        Number of leading iterations during which capCommandedAngle clips phi
        (default 2, matching PHI_CAPS_DEG = (30°, 45°)).  These same
        iterations get the theta offset described below.
    motorMarginMm : float
        Motor-position-error margin (mm) added on top of rDot when computing
        the per-iteration theta offset (default 50 µm).

    Returns
    -------
    thetaStart, phiStart : ndarray (nCobras,)
        Local theta/phi at the dot for each dot cobra; 0 for non-dot cobras.
    phiRamp : ndarray (nIter, nCobras)
        Cumulative phi delta from phiStart.
    thetaRamp : ndarray (nIter, nCobras)
        Cumulative theta delta from thetaStart.  Non-zero for the first
        ``capIters`` iterations of each dot cobra to keep the tip clear of
        the dot while phi is still capped (see note).
    dotGeom : dict
        Keys: phiCenter, halfDot, direction, phiMin, phiMax, thetaOffset.

    Note
    ----
    During the cap iterations, capCommandedAngle clips phi well below
    phiCenter — the arm is more open than the angle that places the tip
    on the dot, so holding theta = thetaDot risks the tip target landing
    inside the dot footprint.  We offset theta by ±Δθ(phi_cap) where
    Δθ(phi) = (rDot + motorMargin) / D(phi) and D(phi) is the tip-to-base
    distance at the capped phi.  This guarantees the commanded tip target
    is at least one dot-radius away from the dot centre, with extra
    margin for motor-position error.  See ``thetaOffsetForPhi``.

    Sign of the offset defaults to +; we flip to − when +offset would push
    theta past the CW hard stop margin, and to 0 if neither sign fits.
    """
    nCobras = len(cc.allCobras)
    thetaStart = np.zeros(nCobras)
    phiStart = np.zeros(nCobras)
    phiRamp = np.zeros((nIter, nCobras))
    thetaRamp = np.zeros((nIter, nCobras))

    thetaStartAll, phiCenter, phiMin, phiMax, phiEnter, direction, halfDot = \
        computeDotAngles(cc)

    phiInDot = computePhiAtFraction(phiCenter, halfDot, direction, 0.1)
    phiStartAll = computePhiStart(phiInDot, direction)

    ramp = computePhiRamp(
        phiStartAll[dotCobras], phiEnter[dotCobras],
        phiInDot[dotCobras], direction[dotCobras], nIter=nIter)

    thetaStart[dotCobras] = thetaStartAll[dotCobras]
    phiStart[dotCobras] = phiStartAll[dotCobras]
    phiRamp[:, dotCobras] = ramp

    # ── theta offset during the cap iterations ────────────────────────────
    # Per-iter offset based on the actual capped phi at that iteration:
    # the lever arm D(phi) is shorter when the arm is more closed (smaller
    # phi), so the required theta sweep is larger.
    thetaRange = (cc.calibModel.tht1 - cc.calibModel.tht0 + np.pi) % (2*np.pi) + np.pi
    thetaMargin = np.deg2rad(15.0)
    L1 = cc.calibModel.L1
    L2 = cc.calibModel.L2
    rDot = sgfm.rDot.to_numpy()

    nCap = min(capIters, nIter)
    iterPhiCapsRad = np.deg2rad(PHI_CAPS_DEG)
    iterOffsetMag = np.zeros((nCap, nCobras))
    for j in range(nCap):
        iterOffsetMag[j] = thetaOffsetForPhi(L1, L2, rDot, iterPhiCapsRad[j],
                                              motorMarginMm=motorMarginMm)

    # Sign choice: default +; flip to − if +max-iter offset clears the CW
    # stop; leave 0 if neither sign keeps both iterations inside margins.
    maxOffset = iterOffsetMag.max(axis=0)
    plusOk  = (thetaStartAll + maxOffset) <= (thetaRange - thetaMargin)
    minusOk = (thetaStartAll - maxOffset) >= thetaMargin
    sign = np.zeros_like(thetaStartAll)
    sign[plusOk]            =  1.0
    sign[~plusOk & minusOk] = -1.0

    for j in range(nCap):
        thetaRamp[j, dotCobras] = (sign * iterOffsetMag[j])[dotCobras]
    thetaOffset = sign * maxOffset   # for diagnostic / dotGeom output

    dotGeom = dict(phiCenter=phiCenter, halfDot=halfDot, direction=direction,
                   phiMin=phiMin, phiMax=phiMax, thetaOffset=thetaOffset)

    return thetaStart, phiStart, phiRamp, thetaRamp, dotGeom


def buildCommandedAngle(thetaStart, phiStart, phiRamp, thetaRamp):
    thetasFull = np.zeros_like(phiRamp)
    phisFull = np.zeros_like(thetaRamp)

    iteration = phiRamp.shape[0]
    for j in range(iteration):
        thetasFull[j] = thetaStart + thetaRamp[j]
        phisFull[j] = phiStart + phiRamp[j]

    return thetasFull, phisFull


def capCommandedAngle(thetaStart, phiStart, phiRamp, thetaRamp, iterPhiCapsDeg=(30.0, 45.0)):
    thetasFull, phisFull = buildCommandedAngle(thetaStart, phiStart, phiRamp, thetaRamp)
    thetasFullSafe = thetasFull.copy()
    phisFullSafe = phisFull.copy()
    iteration = phiRamp.shape[0]

    for j in range(iteration):
        if j < len(iterPhiCapsDeg):
            phiCap = np.deg2rad(iterPhiCapsDeg[j])
            phisFullSafe[j] = np.minimum(phisFull[j], phiCap)

    phiRampOffset = phisFullSafe - phisFull
    phiRamp += phiRampOffset
    return thetaStart, phiStart, phiRamp, thetaRamp


def buildSafeRamp(cc, dotCobras, nIter):
    thetaStart, phiStart, phiRamp, thetaRamp, dotGeom = buildDotRamp(cc, dotCobras, nIter)
    thetaStart, phiStart, phiRamp, thetaRamp = capCommandedAngle(thetaStart, phiStart, phiRamp, thetaRamp)
    return thetaStart, phiStart, phiRamp, thetaRamp, dotGeom


def computePhiAtFraction(phiCenter, halfDot, direction, fraction):
    """Phi angle at a given fractional position through the dot.

    fraction=0   → phiEnter (approach entry edge)
    fraction=0.5 → phiCenter (dot center)
    fraction=1   → phiExit  (approach exit edge)

    Works for both CCW (+1) and CW (-1) approaches.

    Parameters
    ----------
    phiCenter, halfDot, direction : ndarray (nCobras,)
    fraction : float  in [0, 1]

    Returns
    -------
    phi : ndarray (nCobras,)
    """
    # For CCW: entry=phiMin=phiCenter-halfDot, exit=phiMax=phiCenter+halfDot
    # For CW:  entry=phiMax=phiCenter+halfDot, exit=phiMin=phiCenter-halfDot
    # Unified: phi = phiCenter - direction*halfDot + direction*2*halfDot*fraction
    #               = phiCenter + direction*halfDot*(2*fraction - 1)
    return phiCenter + direction * halfDot * (2 * fraction - 1)


def computePhiRamp(phiStart, phiEnter, phiInDot, direction, nIter,
                   edgeMarginDeg=3.0):
    """Build the (nIter, nCobras) phi delta array for the dot ramp.

    Two-phase schedule:
      rows 0 .. nIter-2 : uniform linear ramp stopping edgeMargin before entry edge
                          phiRamp[j] = j * phiStep,  j = 0 .. nIter-2
                          phiStep = (phiRampEnd - phiStart) / (nIter - 2)
                          phiRampEnd = phiEnter - direction * edgeMargin
      row  nIter-1      : jump to phiInDot (just inside entry edge, fraction=0.1)
                          phiRamp[nIter-1] = phiInDot - phiStart

    The discontinuity at the last step is intentional: the arm approaches
    smoothly to within edgeMargin of the dot, then is commanded in one step
    to cross the entry edge and land at fraction=0.1 inside the dot.

    Parameters
    ----------
    phiStart  : ndarray (nCobras,)  ramp start phi (= phis[dotCobras])
    phiEnter  : ndarray (nCobras,)  phi at dot entry edge (phiMin CCW, phiMax CW)
    phiInDot  : ndarray (nCobras,)  target phi inside dot (fraction=0.1)
    direction : ndarray (nCobras,) int  +1 CCW, -1 CW
    nIter     : int  total iterations (= tries).  Must be >= 3.
    edgeMarginDeg : float  degrees before entry edge where linear ramp ends.

    Returns
    -------
    phiRamp : ndarray (nIter, nCobras)
        Cumulative phi delta from phiStart.  Zero for science cobras.
    """
    edgeMargin = np.deg2rad(edgeMarginDeg)
    phiRampEnd = phiEnter - direction * edgeMargin  # 3° before entry edge
    phiStep = (phiRampEnd - phiStart) / (nIter - 2)  # signed, nIter-1 uniform steps

    j_arr = np.arange(nIter - 1)
    ramp = j_arr[:, None] * phiStep[None, :]  # (nIter-1, nCobras)
    jump = (phiInDot - phiStart)[None, :]  # (1, nCobras)
    return np.concatenate([ramp, jump], axis=0)  # (nIter, nCobras)


def computePhiStart(phiInDot, direction, phiFloor=np.radians(15.0),
                    minRangeDeg=30.0):
    """Compute per-cobra ramp start phi.

    Guarantees at least minRange radians of ramp travel where possible.

    CCW (+1): phiStart = max(phiFloor, phiInDot - minRange)
              Travel = phiInDot - phiStart (may be < minRange if phiInDot is
              close to phiFloor — constrained by the hard stop, nothing we can do).

    CW  (-1): phiStart = phiInDot + max(phiInDot - phiFloor, minRange)
              Travel = phiStart - phiInDot >= minRange always.

    Parameters
    ----------
    phiInDot : ndarray (nCobras,)
        Target phi at the end of the ramp (just inside entry edge).
    direction : ndarray (nCobras,) int
        +1 CCW, -1 CW.
    phiFloor : float
        Minimum starting phi (hard-stop margin, radians). Default 15°.
    minRangeDeg : float
        Minimum guaranteed ramp travel in degrees. Default 30°.

    Returns
    -------
    phiStart : ndarray (nCobras,)
    """
    minRange = np.deg2rad(minRangeDeg)
    phiStart = np.maximum(phiFloor, phiInDot - minRange)  # CCW default

    cwMask = direction < 0
    cw_travel = np.maximum(phiInDot[cwMask] - phiFloor, minRange)
    phiStart[cwMask] = phiInDot[cwMask] + cw_travel
    return phiStart


def fitPhiSpeed(moves, localIdx, nFit=4):
    """Estimate phi angular speed (rad/step) for a dot cobra.

    Uses the last nFit *truly-detected* iterations (moves['detected']==True) to
    compute nFit-1 consecutive Δphi/phiSteps ratios, then returns the median.
    Iterations where the cobra was not detected by MCS must be skipped because
    cobra_match falls back to the dot centre, and positionsToAngles(dot_centre)
    returns phiCenter — a pure artifact, not a real measurement.

    Parameters
    ----------
    moves : structured ndarray, shape (nDotCobras, nIter)
        Slice of the moves array for dot cobras only.  Required fields:
        'phiAngle', 'phiSteps', 'detected'.
    localIdx : int
        Index within the dot-cobra slice (0..nDotCobras-1).
    nFit : int
        Number of detected iterations to use. Default 4.

    Returns
    -------
    speed : float
        Median |Δphi / phiSteps| in rad/step.  NaN if insufficient data.
    """
    detected = moves['detected'][localIdx]
    phi      = moves['phiAngle'][localIdx]
    steps    = moves['phiSteps'][localIdx]

    visIdx = np.where(detected)[0]
    if len(visIdx) < 2:
        return np.nan

    last = visIdx[-min(nFit, len(visIdx)):]
    dphi = np.diff(phi[last])
    dstep = steps[last[1:]]

    valid = dstep != 0
    if not np.any(valid):
        return np.nan

    return float(np.median(np.abs(dphi[valid] / dstep[valid])))


def computeBlindSteps(speed, direction, halfDot,
                      fromFraction=0.1, toFraction=0.4):
    """Open-loop phi step count to push a hidden cobra from fromFraction
    to toFraction inside its dot.

    Hidden cobras are *not* reliably measured: cobra_match falls back to
    the dot centre, positionsToAngles returns phiCenter, and any IK that
    uses it generates step counts that yo-yo the cobra in/out of the dot.
    The recorded phiAngle and phiSteps history is therefore polluted from
    the moment the cobra first goes hidden.

    Instead of trying to recover the cobra's actual current phi from that
    history, we assume it is at fromFraction (the iter-15 ramp target,
    where the convergence loop last commanded it) and step the *fixed*
    delta to toFraction:

        Δphi  = direction · (toFraction − fromFraction) · 2 · halfDot
        steps = Δphi / speed

    Sign comes from direction · (Δfraction); speed is the positive
    rad/step magnitude from fitPhiSpeed.

    Parameters
    ----------
    speed : float
        Positive rad/step estimate from fitPhiSpeed (NaN → return 0).
    direction : int
        +1 CCW (opening), −1 CW (closing).
    halfDot : float
        Half-angle subtended by the dot at the elbow (radians).
    fromFraction, toFraction : float
        Assumed starting depth and target depth inside the dot.
        0 = entry edge, 0.5 = centre, 1 = exit edge.

    Returns
    -------
    steps : int
        Signed phi step count (positive = CCW/opening).
    """
    if np.isnan(speed) or speed == 0:
        return 0
    deltaPhi = direction * (toFraction - fromFraction) * 2 * halfDot
    return int(round(deltaPhi / speed))


def estimateMotorMapSpeed(cc, cIds, phiAngles, direction):
    """Estimate phi angular speed (rad/step) from the calibrated motor map.

    Evaluated at the region of the motor map that was actually characterized:
      CCW approach (+1) → evaluate at phiAngles (typically phiMax, arm extended)
      CW  approach (-1) → evaluate at phiAngles (typically phiMin, arm retracted)

    Parameters
    ----------
    cc : CobraCoach
    cIds : array-like of int
        Cobra indices.
    phiAngles : array-like of float
        Phi angles at which to evaluate the motor map (radians).
    direction : array-like of int
        +1 CCW, -1 CW per cobra.

    Returns
    -------
    speed : ndarray (len(cIds),)
        Motor-map rad/step estimate per cobra.
    """
    cIds = np.asarray(cIds)
    phiAngles = np.asarray(phiAngles)
    direction = np.asarray(direction)
    cm = cc.calibModel
    speed = np.zeros(len(cIds))

    for i, (cId, phi, dirn) in enumerate(zip(cIds, phiAngles, direction)):
        if dirn > 0:
            stepArr = cm.posPhiSlowSteps[cId]
        else:
            stepArr = cm.negPhiSlowSteps[cId]
        angArr = cm.phiOffsets[cId]

        if len(angArr) < 2 or stepArr[-1] == 0:
            speed[i] = np.nan
            continue

        # rad/step at phi: local derivative via finite difference on the map
        phi_clipped = np.clip(phi, angArr[0], angArr[-1])
        idx = np.searchsorted(angArr, phi_clipped)
        idx = np.clip(idx, 1, len(angArr) - 1)

        dAng = angArr[idx] - angArr[idx - 1]
        dStep = stepArr[idx] - stepArr[idx - 1]
        speed[i] = abs(dAng / dStep) if dStep != 0 else np.nan

    return speed


def blindMoveHiddenCobras(cc, dotGlobalIdx, filteredGoodIdx, moves, dotGeom,
                          fromFraction=0.1, toFraction=0.4, cmd=None):
    """Open-loop step push of hidden dot cobras toward toFraction inside the dot.

    Called after the convergence loop in moveToPfsDesign.  Cobras whose spot has
    disappeared from MCS (cc.cobraInfo['detected'] == False) are at or just
    behind the dot rim.  Once hidden, neither the cobra's measured phi nor the
    closed-loop step counts can be trusted (cobra_match's dot-centre fallback
    drives the IK to send wild yo-yo moves).  We assume each hidden cobra is at
    fromFraction (the iter-15 ramp target) and step the fixed delta to
    toFraction in the correct CCW/CW direction.

    Parameters
    ----------
    cc : CobraCoach
        Source of cc.cobraInfo['detected'], cc.allCobras, cc.pfi.moveSteps.
    dotGlobalIdx : ndarray of int
        Global indices of cobras that participated in the dot ramp.
    filteredGoodIdx : ndarray of int
        Global indices in the order they appear along axis-1 of ``moves``.
    moves : structured ndarray, shape (1, len(filteredGoodIdx), nIter)
        moveThetaPhi history (used by fitPhiSpeed for the rad/step estimate).
    dotGeom : dict
        Output of buildDotRamp; must carry halfDot and direction.
    fromFraction : float
        Assumed depth at which each hidden cobra sits before the blind move.
        Defaults to 0.1 — the iter-15 ramp target inside the dot.
    toFraction : float
        Target depth.  Default 0.4 — well past the rim, comfortably hidden.
    cmd : optional
        Command-handle for inform/warn messages.

    Returns
    -------
    nCommanded : int
        Number of cobras that received a blind step command.
    """
    detected     = cc.cobraInfo['detected']
    hiddenGlobal = dotGlobalIdx[~detected[dotGlobalIdx]]

    if cmd is not None:
        cmd.inform(f'text="blind move: {len(hiddenGlobal)}/{len(dotGlobalIdx)} '
                   f'hidden dot cobras  fraction {fromFraction}→{toFraction}"')

    if len(hiddenGlobal) == 0:
        return 0

    halfDot   = dotGeom['halfDot']
    direction = dotGeom['direction']

    idxMap = {int(gc): i for i, gc in enumerate(np.asarray(filteredGoodIdx))}
    cobrasToStep = []
    phiStepsList = []
    nSpeedNan = 0
    for cIdGlobal in hiddenGlobal:
        iLocal = idxMap.get(int(cIdGlobal))
        if iLocal is None:
            continue
        speed = fitPhiSpeed(moves[0], iLocal)
        if np.isnan(speed) or speed == 0:
            nSpeedNan += 1
            continue
        steps = computeBlindSteps(speed, direction[cIdGlobal], halfDot[cIdGlobal],
                                   fromFraction=fromFraction, toFraction=toFraction)
        if steps == 0:
            continue
        cobrasToStep.append(cc.allCobras[cIdGlobal])
        phiStepsList.append(int(steps))

    nCommanded = len(cobrasToStep)
    if nCommanded:
        # Use cc.pfi.moveSteps (pure FPGA, no MCS exposure) — this is a
        # genuine open-loop blind move, and avoids the per-call MCS frame
        # that cc.moveSteps otherwise triggers via exposeAndExtractPositions.
        # phiFast=True matches the fast-motor-map pulsing used during the
        # slow-ramp moveThetaPhi from which fitPhiSpeed was calibrated.
        thetaSteps = np.zeros(nCommanded, dtype=int)
        cc.pfi.moveSteps(cobrasToStep, thetaSteps,
                         np.asarray(phiStepsList, dtype=int),
                         phiFast=True)

    if cmd is not None:
        cmd.inform(f'text="blind move: commanded {nCommanded}/{len(hiddenGlobal)} '
                   f'(skipped {nSpeedNan} with no speed estimate)"')
    return nCommanded
