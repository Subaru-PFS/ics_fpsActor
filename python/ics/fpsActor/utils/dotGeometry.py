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
from ics.cobraCharmer import targetValidation


# Phi cap (deg) per leading "fast" iteration of moveThetaPhi.  Both the phi
# clipping in capCommandedAngle and the matching theta safety offset in
# buildDotRamp use this same schedule.
PHI_CAPS_DEG = (30.0, 45.0)

# Dot-hiding depth schedule, in dot fraction (0 = entry edge, 0.5 = dot centre,
# 1 = exit edge).  The closed-loop phi ramp lands the tip at RAMP_LANDING_FRACTION;
# a single open-loop slow-map blind move (dotMove.blindMoveToDots) then pushes it
# deeper, to each cobra's own measured optimum.
#
# Measured on the 2026-07-24 telescope run (good cobras, flux right after
# moveToPfsDesign): landing at 0.2 with no blind move leaves a median 9.6% residual flux
# (35% of cobras below 5%); the 0.1 -> 0.3 push reached 0.62% (84% below 5%).
#
# The two 13-step across-dot scans give the fraction of cobras below 5% residual flux for
# a PERFECTLY placed tip as
#     0.30 -> 89.8%    0.35 -> 96.8%    0.40 -> 98.4%    0.45 -> 98.3%    0.50 -> 96.7%
# so the obscuration optimum is a broad plateau from about 0.35 to 0.50.
#
# BLIND_TARGET_FRACTION is only the fallback.  Each cobra is aimed at its own measured
# optimum where one exists (dotTargets), because the depth at which a cobra hides best is
# a property of that cobra: the fitted optima span 0.21 to 0.80 with 158 um of real
# cobra-to-cobra spread, against 25 um of measurement scatter.  Interpolating the scan
# curves at each aim point gives the fraction hidden below 1% residual flux as
#     fixed 0.40  89.8%    fixed 0.445  92.0%    fixed 0.50  89.7%    per-cobra  98.8%
# so 0.445 is the best single value and is what an uncalibrated cobra gets.
#
# Optima beyond 0.5 are real and are not clipped: for the 1108 cobras whose measured
# optimum is past the dot centre, aiming there hides 98.7% against 83.3% at 0.40.  What
# the fraction cannot exceed is the range the scan actually covered -- dotTargets rejects
# anything outside it rather than clamping, since a clamped value still asserts a depth
# nobody observed.
RAMP_LANDING_FRACTION = 0.1
BLIND_TARGET_FRACTION = 0.4

# Where the closed-loop ramp stops, just outside the entry edge, before the landing jump.
# A fixed angle is not a fixed standoff -- halfDot varies across the fleet, so 3 degrees
# spans 0.076 to 0.087 of a dot -- and every other depth in this module is a fraction.
# Measured against the detection boundary on the 2026-07-24 scans, -0.075 leaves a median
# margin of 168 um (p5 51 um) and keeps 98.1% of cobras outside it, which matches the
# 99.7% actually still detected at that iteration.  Do not shrink it further: the last
# measured iteration is what the blind move departs from, and below about -0.05 it lands
# inside the scatter of where the edge really is.
RAMP_APPROACH_FRACTION = -0.075

# Bounds on the open-loop blind move.  fitPhiSpeed stays the step estimator — it
# tracks real behaviour better than the calibrated map — but it measures achieved
# motion over the last few near-dot iterations, so a stalled or backlash-dominated
# iteration collapses it and the step count blows up.  BLIND_SPEED_SLACK lets the
# fit run this many times slower than the motor map before the map caps it;
# MAX_BLIND_STEPS is the absolute backstop when the map itself is unusable.
#
# Sized on the 2026-07-24 run: across 6665 real blind-move commands the largest
# legitimate one was 2349 steps (p50 27-109, p99 67-246), and these bounds touch
# 0.2% of commands while leaving p50/p99 unchanged.  They exist for cases like
# cobra 1963, which fitted 4.3e-07 rad/step and asked for 73919 steps —
# pfi.moveSteps silently clipped that to 6000 (~316 deg of phi) and drove it out
# of its dot, fully lit, for the remaining 11 scan steps.
MAX_BLIND_STEPS = 3000
BLIND_SPEED_SLACK = 3.0


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


def buildDotRamp(cc, dotCobras, nIter, capIters=2, motorMarginMm=0.2,
                 landingFraction=RAMP_LANDING_FRACTION):
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
        the per-iteration theta offset (default 200 µm).  Sized to absorb the
        large first-move execution error during the cap iterations, which lands
        ~12-17% of cobras on the dot with the old 50 µm margin even though the
        commanded tip is clear (2026-07-24 run).  Feasibility-checked: the dodge
        still fits between the theta hard stops for every cobra (sign=0 count
        stays 0 up to 0.5 mm).  Effect to be confirmed on the next run alongside
        the fast-map-on-first-iteration fix.
    landingFraction : float
        Dot fraction the last ramp row lands at.  The run-up before it is unchanged,
        so only the length of the final jump across the edge varies.

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

    # The ramp starts a fixed run-up before the nominal landing, whatever depth the
    # last row is asked for.  Anchoring the start to RAMP_LANDING_FRACTION rather than
    # to `landingFraction` keeps it equal to the phiStart the dot design was built from,
    # so the design and the ramp describe the same starting angle.
    phiInDot = computePhiAtFraction(phiCenter, halfDot, direction, RAMP_LANDING_FRACTION)
    phiStartAll = computePhiStart(phiInDot, direction)

    # Where the last row lands.  A scan wants this shallow, to climb the obscuration
    # curve from outside; a sequence that only has to hide wants it deeper, so that a
    # blind move falling short still leaves the cobra behind its dot.
    phiFinal = computePhiAtFraction(phiCenter, halfDot, direction, landingFraction)

    # Stop the closed-loop ramp just outside the entry edge, as a fraction of each
    # cobra's own dot rather than a fixed angle.
    phiRampEnd = computePhiAtFraction(phiCenter, halfDot, direction, RAMP_APPROACH_FRACTION)

    ramp = computePhiRamp(
        phiStartAll[dotCobras], phiRampEnd[dotCobras],
        phiFinal[dotCobras], direction[dotCobras], nIter=nIter)

    thetaStart[dotCobras] = thetaStartAll[dotCobras]
    phiStart[dotCobras] = phiStartAll[dotCobras]
    phiRamp[:, dotCobras] = ramp

    # ── theta offset during the cap iterations ────────────────────────────
    # Per-iter offset based on the actual capped phi at that iteration:
    # the lever arm D(phi) is shorter when the arm is more closed (smaller
    # phi), so the required theta sweep is larger.
    thetaRange = targetValidation.thetaRange(cc.calibModel)
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


def buildSafeRamp(cc, dotCobras, nIter, landingFraction=RAMP_LANDING_FRACTION):
    thetaStart, phiStart, phiRamp, thetaRamp, dotGeom = buildDotRamp(
        cc, dotCobras, nIter, landingFraction=landingFraction)
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


def computePhiRamp(phiStart, phiRampEnd, phiInDot, direction, nIter):
    """Build the (nIter, nCobras) phi delta array for the dot ramp.

    Two-phase schedule:
      rows 0 .. nIter-2 : uniform linear ramp stopping at phiRampEnd, just outside the
                          entry edge
                          phiRamp[j] = j * phiStep,  j = 0 .. nIter-2
                          phiStep = (phiRampEnd - phiStart) / (nIter - 2)
      row  nIter-1      : jump to phiInDot (just inside entry edge, fraction=0.1)
                          phiRamp[nIter-1] = phiInDot - phiStart

    The discontinuity at the last step is intentional: the arm approaches
    smoothly to within edgeMargin of the dot, then is commanded in one step
    to cross the entry edge and land at fraction=0.1 inside the dot.

    Parameters
    ----------
    phiStart   : ndarray (nCobras,)  ramp start phi (= phis[dotCobras])
    phiRampEnd : ndarray (nCobras,)  where the linear ramp stops, just outside the edge
    phiInDot  : ndarray (nCobras,)  target phi inside dot (fraction=0.1)
    direction : ndarray (nCobras,) int  +1 CCW, -1 CW
    nIter     : int  total iterations (= tries).  Must be >= 3.

    Returns
    -------
    phiRamp : ndarray (nIter, nCobras)
        Cumulative phi delta from phiStart.  Zero for science cobras.
    """
    if nIter < 3:
        raise ValueError(f'dot ramp needs nIter >= 3, got {nIter}: the schedule '
                         'reserves one row for the landing jump and divides the '
                         'linear phase by (nIter - 2)')

    phiStep = (phiRampEnd - phiStart) / (nIter - 2)  # signed, nIter-1 uniform steps

    j_arr = np.arange(nIter - 1)
    ramp = j_arr[:, None] * phiStep[None, :]  # (nIter-1, nCobras)
    jump = (phiInDot - phiStart)[None, :]  # (1, nCobras)
    return np.concatenate([ramp, jump], axis=0)  # (nIter, nCobras)


def computePhiStart(phiInDot, direction, phiFloor=np.radians(15.0),
                    minRangeDeg=30.0):
    """Compute per-cobra ramp start phi.

    The ramp starts minRange away from the landing angle, on the far side from
    the dot, so both approach directions get the same run-up:

        CCW (+1): phiStart = phiInDot - minRange   (approaches from below)
        CW  (-1): phiStart = phiInDot + minRange   (approaches from above)

    The phiFloor clamp is a safety net that the direction choice already makes
    unreachable: computeDotAngles only selects CCW when
    phiMin >= phiFloor + minRange, which is exactly the condition for
    phiInDot - minRange >= phiFloor.

    The run-up is deliberately NOT scaled by how far phiInDot sits above the
    floor.  An earlier version did, using
        phiStart = phiInDot + max(phiInDot - phiFloor, minRange)
    for the CW branch.  Because phiInDot - phiFloor is a median 51 deg for CW
    cobras, that branch won for 99.8% of them and gave CW a 40 deg ramp span
    against CCW's 19 deg -- so CW cobras were commanded twice the per-iteration
    step for no reason.  Measured on the 2026-07-24 dataset, execution error is
    a fixed ~17% of the commanded step in both directions, so the oversized
    run-up doubled the CW residual (45 um against 24 um for CCW at the same
    iteration) and cost ~14 points of arrival within 100 um.

    Parameters
    ----------
    phiInDot : ndarray (nCobras,)
        Target phi at the end of the ramp (just inside entry edge).
    direction : ndarray (nCobras,) int
        +1 CCW, -1 CW.
    phiFloor : float
        Minimum starting phi (hard-stop margin, radians). Default 15°.
    minRangeDeg : float
        Ramp travel before the dot, in degrees. Default 30°.

    Returns
    -------
    phiStart : ndarray (nCobras,)
    """
    minRange = np.deg2rad(minRangeDeg)
    return np.maximum(phiFloor, phiInDot - direction * minRange)


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
                      fromFraction=0.1, toFraction=0.4, maxSteps=None):
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
    maxSteps : float or None
        Upper bound on |steps|.  fitPhiSpeed measures *achieved* motion over the
        last few near-dot iterations; a stalled or backlash-dominated iteration
        collapses the estimate and the step count blows up.  On 2026-07-24 cobra
        1963 fitted 4.3e-07 rad/step (2000x below the fleet median), which asked
        for 73919 steps — pfi.moveSteps silently clipped that to 6000 (~316 deg
        of phi) and drove the cobra clean out of its dot for the rest of the run.
        None disables the bound (the pre-fix behaviour).

    Returns
    -------
    steps : int
        Signed phi step count (positive = CCW/opening), clamped to maxSteps.
    """
    if np.isnan(speed) or speed == 0:
        return 0
    deltaPhi = direction * (toFraction - fromFraction) * 2 * halfDot
    steps = deltaPhi / speed

    if maxSteps is not None and np.isfinite(maxSteps) and abs(steps) > maxSteps:
        steps = np.sign(steps) * maxSteps

    return int(round(steps))


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
