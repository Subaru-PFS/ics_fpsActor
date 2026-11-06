"""Fit, validate and select the per-cobra calibration of the two blind moves.

The measurements are the rows of opdb `cobra_blind_move`, one per cobra and turned-dot
visit; everything here is computed from their columns alone, including the dot chord,
so a calibration can be refitted or checked without the run directories or the geometry.

A calibration is written once (`blind_move_calib`, `cobra_blind_move_gain`) and becomes
the default when it is judged good on aimed runs (`blind_move_calib_validation`): the
default is the calibration of the newest validation row.  Nothing is ever updated.

Two models.  'gain-lsq' fits each gain by least squares over runs at several depths.
'operating-point' measures the gains of the moves production makes -- the landing to one
depth, the push from there to one target -- on runs at that point aimed through a prior
calibration: each gain is the robust mean of a cobra's runs, so nothing is extrapolated
and the form of the response does not matter.  The prior aims the runs and is compared
with them, but does not enter the gains: they wander from run to run and from night to
night by more than a prior from another time can predict.

A gain is the measured phi travel of a move over the travel its steps were sized for,
both from the measured start.  Rows record the commanded angles (phi_landing_command,
phi_push_start, phi_push_command); rows from before those columns existed fall back to
the depth targets, which is what their moves were sized to.
"""
import json

import numpy as np
import pandas as pd
from sqlalchemy import exc, text

from ics.fpsActor.utils import dotGeometry, dotState

MODEL = 'gain-lsq'
"""Gain-only model, least squares through the origin."""

MODEL_VERSION = 2
"""Version 2 accepts gains within `dotState.GAIN_BOUNDS`; version 1 accepted (0.5, 2.0)."""

OPERATING_POINT_MODEL = 'operating-point'
"""Gains measured on runs at the production landing and target, aimed through a prior
calibration."""

OPERATING_POINT_VERSION = 1

OPERATING_POINT = dict(minRuns=3, noiseDof=8.0, huber=2.5, reject=5.0, regime=4.0,
                       regimeJump=0.25, stallGain=0.3, minSpan=0.4, noiseFloor=0.01)
"""Defaults of `fitOperatingPoint`:

minRuns    runs of each move a cobra needs before its gains are used, counting those
           before a change of regime
noiseDof   degrees of freedom of the fleet's run-to-run scatter in a cobra's noise
huber      standardised residual past which a run is pulled back to that distance
reject     standardised residual past which a run is dropped
regime     standardised step between the earlier and later runs of a cobra that, with
           regimeJump, marks a change of regime: its gain is then that of the later runs
regimeJump that step as a fraction of the gain.  Both are needed: a gain that drifts over a
           session is significant too, but its mean is still the best value it has
stallGain  landing gain below which a run counts as a stall; two stalls mark the cobra
minSpan    shortest commanded travel used, as a fraction of the fleet median of its move
noiseFloor smallest per-run noise of a gain, so that identical runs do not weigh infinitely
"""

LANDING_COMMAND_RANGE = (0.05, 0.7)
"""Commanded landing depth an aim may ask for, in dot fraction: from the ramp approach plus
0.05 to 0.7.  The upper end keeps a cobra whose low landing gain has gone stale, and which
moves the map's travel instead, well short of the exit edge."""

RAMP_RESIDUAL_CUT = 0.100
"""Largest distance from target before the landing, mm, a move is fitted with.  Past it
the cobra did not start the landing where the ramp put it."""

MIN_RUNS = 3
"""Passing runs a cobra needs before its gains are used."""

SLOW_LANDING_GAIN = 0.5
"""Landing gain below which the validation reports a cobra's landings apart."""

AIM_TOLERANCE = 1e-3
"""Largest difference, in dot fraction, between a recorded and a recomputed command for a
visit to count as aimed by a calibration."""

VALIDATION_CRITERIA = dict(maxBias=0.015, maxSpread=0.035, minWithin05=0.80, minCoverage=0.90,
                           maxLandingBias=0.01, maxLandingSpread=0.02, minLandingWithin05=0.95,
                           maxShortLandings=0.005, maxRegression=0.002)
"""Defaults of the validation, over all aimed visits:

maxBias, maxSpread, minWithin05   |median|, 16-84% half-width and fraction within 0.05 of
                                  the depth reached less dot_target
minCoverage                       fraction of the dot cobras measured
maxLandingBias, maxLandingSpread, the same for the landing depth less dot_landing of the
minLandingWithin05                calibrated cobras, judged when the landing was aimed
maxShortLandings                  fraction of the landings short of dot_landing by more
                                  than 0.08, which is where cobras stay in view
maxRegression                     how much worse than the reference calibration the rms
                                  counterfactual error may be, landing and final alike,
                                  errors less their median and clipped at `REGRESSION_CLIP`
"""

REGRESSION_CLIP = 0.1
"""Dot fraction the counterfactual errors are clipped at, once less their median, when two
calibrations are compared: a few cobras far from any gain would otherwise decide the
comparison, and the common mode of the night is judged by maxBias."""

MEASUREMENT_COLUMNS = ['pfs_visit_id', 'cobra_id', 'dot_landing', 'dot_target', 'dot_command',
                       'phi_landing_target', 'phi_push_target', 'phi_start', 'phi_landed',
                       'phi_pushed', 'ramp_residual', 'phi_landing_command', 'phi_push_start',
                       'phi_push_command', 'blind_move_calib_id']


def chordOf(meas):
    """Signed phi per unit dot fraction, radians, of each measurement row.

    NaN where the push was commanded no deeper than the landing, which defines no chord.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        chord = ((meas.phi_push_target - meas.phi_landing_target)
                 / (meas.dot_command - meas.dot_landing)).to_numpy(dtype=float)
    return np.where(np.isfinite(chord) & (chord != 0), chord, np.nan)


def _legacy(meas):
    """Rows written before the commanded angles were recorded: all three of them NULL."""
    names = ('phi_landing_command', 'phi_push_start', 'phi_push_command')
    return np.all([meas[n].isna().to_numpy() if n in meas else np.ones(len(meas), dtype=bool)
                   for n in names], axis=0)


def moveTravels(meas):
    """Commanded and measured phi travel of the landing and the push of each row, radians.

    The landing is commanded from the measured phi_start to phi_landing_command, the push
    from phi_push_start, the estimate it was sized from, to phi_push_command, where its
    steps reach at gain 1.  The measured travels start from the measured phi_start and
    phi_landed.  On rows from before the commanded angles were recorded, the landing was
    commanded to phi_landing_target and the push from there to phi_push_target.

    Returns
    -------
    `pandas.DataFrame`
        Columns x1, y1 (landing commanded, measured) and x2, y2 (push), indexed as `meas`.
    """
    legacy = _legacy(meas)

    def angle(name, fallback):
        recorded = meas[name].to_numpy(dtype=float) if name in meas else np.nan
        return np.where(legacy, meas[fallback].to_numpy(dtype=float), recorded)

    landingCommand = angle('phi_landing_command', 'phi_landing_target')
    pushStart = angle('phi_push_start', 'phi_landing_target')
    pushCommand = angle('phi_push_command', 'phi_push_target')
    start, landed, pushed = (meas[c].to_numpy(dtype=float) for c in ('phi_start', 'phi_landed',
                                                                    'phi_pushed'))
    return pd.DataFrame(dict(x1=landingCommand - start, y1=landed - start,
                             x2=pushCommand - pushStart, y2=pushed - landed), index=meas.index)


def usable(meas, cut=RAMP_RESIDUAL_CUT):
    """Rows whose two moves were measured and started where the ramp put them."""
    travels = moveTravels(meas)
    measured = travels.notna().all(axis=1)
    return (measured & (meas.ramp_residual <= cut)).to_numpy()


def boundedGains(gains):
    """`gains` where `dotState.usableGain` believes them, else `dotState.DEFAULT_GAIN`."""
    gains = np.asarray(gains, dtype=float)
    lo, hi = dotState.GAIN_BOUNDS
    with np.errstate(invalid='ignore'):
        return np.where(np.isfinite(gains) & (gains >= lo) & (gains <= hi), gains,
                        dotState.DEFAULT_GAIN)


def landingCommands(landing, approach, landingGain, calibrated):
    """Depth to command the landing to, per cobra, so that it lands at `landing`.

    The landing runs from about the ramp approach, `approach`, and travels landingGain
    times what it is commanded:

        command = approach + (landing - approach) / landingGain

    with the gain bounded as the tracker bounds it.  A cobra not calibrated is commanded to
    `landing`.  Depths in dot fraction; arrays broadcast.

    Returns
    -------
    `numpy.ndarray`
        Clipped to `LANDING_COMMAND_RANGE`, its lower end counted from `approach`.
    """
    command = approach + (landing - approach) / boundedGains(landingGain)
    command = np.clip(command, approach + LANDING_COMMAND_RANGE[0], LANDING_COMMAND_RANGE[1])
    return np.where(np.asarray(calibrated, dtype=bool), command, landing)


def aimedCommand(wanted, landing, phiStart, phiLandingTarget, chord, landingGain, targetGain):
    """Depth to command so that a cobra arrives at the depth `wanted`.

    The landing carries the cobra from the measured `phiStart` towards the landing depth,
    and the push is commanded as a fraction of the chord from that landing depth:

        landed  = phiStart + landingGain * (phiLandingTarget - phiStart)
        command = landing + (phiAt(wanted) - landed) / (targetGain * chord)

    with phiAt(wanted) = phiLandingTarget + (wanted - landing) * chord.  Gains of 1.0 give
    `wanted` back.  Every argument broadcasts; angles in radians, depths in dot fraction.

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`)
        Depth to command, and the mask of cobras aimed; a cobra without a finite command
        in (0, 1.5) keeps `wanted`.
    """
    wanted = np.asarray(wanted, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        phiWanted = phiLandingTarget + (wanted - landing) * chord
        landed = phiStart + landingGain * (phiLandingTarget - phiStart)
        command = landing + (phiWanted - landed) / (targetGain * chord)
        aimed = np.isfinite(command) & (command > 0.0) & (command < 1.5)
    return np.where(aimed, command, wanted), aimed


def fitGains(meas, exclude=None, cut=RAMP_RESIDUAL_CUT, minRuns=MIN_RUNS):
    """Landing and push gain per cobra.

    Each gain is the measured phi change over the commanded one, see `moveTravels`.  The
    push's measured change starts from phi_landed, so the two gains are independent.  On
    rows recording no push command, a push clipped at `dotGeometry.MAX_BLIND_STEPS` reads
    as a low push gain.

    Parameters
    ----------
    meas : `pandas.DataFrame`
        `cobra_blind_move` rows, any number of visits.
    exclude : `int`, optional
        Visit left out of the fit.
    cut : `float`
    minRuns : `int`

    Returns
    -------
    `pandas.DataFrame`
        Indexed by cobra_id: landing_gain, target_gain, n_runs, landing_scatter,
        target_scatter, calibrated.
    """
    use = meas[usable(meas, cut=cut) & (meas.pfs_visit_id != exclude).to_numpy()]
    travels = moveTravels(use)
    x1, y1, x2, y2 = travels.x1, travels.y1, travels.x2, travels.y2
    with np.errstate(invalid='ignore', divide='ignore'):
        per = pd.DataFrame(dict(cobra_id=use.cobra_id, xx1=x1 * x1, xy1=x1 * y1, xx2=x2 * x2,
                                xy2=x2 * y2, g1=y1 / x1, g2=y2 / x2))
    grouped = per.groupby('cobra_id')
    sums = grouped[['xx1', 'xy1', 'xx2', 'xy2']].sum()

    out = pd.DataFrame(index=sums.index)
    out['landing_gain'] = sums.xy1 / sums.xx1
    out['target_gain'] = sums.xy2 / sums.xx2
    out['n_runs'] = grouped.size()
    out['landing_scatter'] = grouped.g1.std()
    out['target_scatter'] = grouped.g2.std()
    out['calibrated'] = ((out.n_runs >= minRuns)
                         & out.landing_gain.between(*dotState.GAIN_BOUNDS)
                         & out.target_gain.between(*dotState.GAIN_BOUNDS))
    return out


def _robustVariance(values):
    """Squared 1.4826 MAD of the finite `values`; NaN with fewer than three."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    return float((1.4826 * np.median(np.abs(values - np.median(values)))) ** 2)


def _medianChi2Ratio(dof):
    """Median of a chi-squared variable with `dof` degrees of freedom, over `dof`
    (Wilson-Hilferty): what the median of sample variances underestimates the variance by."""
    return (1 - 2 / (9 * dof)) ** 3


def _regimeSplit(y, noiseVar):
    """The split of a cobra's runs, in time order, with the largest standardised difference
    of means, splits leaving at least two runs on each side.

    Returns
    -------
    (`float`, `float`, `int`)
        That standardised difference, the difference over the mean of all runs, and the
        number of runs before the split.
    """
    m = len(y)
    best = (0.0, 0.0, 0)
    for k in range(2, m - 1):
        step = abs(y[:k].mean() - y[k:].mean())
        z = step / np.sqrt(noiseVar * (1 / k + 1 / (m - k)))
        if z > best[0]:
            best = (z, step / abs(y.mean()), k)
    return best


def _moveRatios(meas, p, cut):
    """Measured over commanded travel of the usable runs of each move, in time order.

    Returns
    -------
    `dict`
        'landing' and 'push': (ratios, cobra ids, visits).
    """
    meas = meas.sort_values(['pfs_visit_id', 'cobra_id'])
    travels = moveTravels(meas)
    started = (meas.ramp_residual <= cut).to_numpy()
    out = {}
    for move, x, y in (('landing', travels.x1, travels.y1), ('push', travels.x2, travels.y2)):
        x, y = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & started
        span = np.median(np.abs(x[finite])) if finite.any() else np.nan
        valid = finite & (np.abs(x) >= p['minSpan'] * span)
        out[move] = (y[valid] / x[valid], meas.cobra_id.to_numpy()[valid],
                     meas.pfs_visit_id.to_numpy()[valid])
    return out


def _fitMove(ratio, cobraIds, p):
    """Robust mean of each cobra's runs of one move.

    Runs further than `reject` standard deviations from the cobra's median are dropped and
    those further than `huber` pulled back, with three runs or more; with fewer nothing
    tells which run is wrong.  A cobra's per-run noise pools the fleet's, corrected for the
    bias of a median of sample variances, with its own residuals; the distance a run is
    pulled back from uses its median absolute deviation instead, which the run itself
    does not inflate.

    Returns
    -------
    (`pandas.DataFrame`, `float`)
        By cobra: gain, noise (per-run variance), n (runs used), rejected, split (runs
        before a change of regime, dropped); and the fleet's per-run noise, as a gain.
    """
    runs = {c: v.to_numpy() for c, v in pd.Series(ratio).groupby(cobraIds)}
    counts = [len(v) for v in runs.values() if len(v) >= 3]
    sampleVars = [np.var(v, ddof=1) for v in runs.values() if len(v) >= 3]
    fleetNoise = (float(np.median(sampleVars)) / _medianChi2Ratio(np.median(counts) - 1)
                  if sampleVars else 0.05 ** 2)
    fleetNoise = max(fleetNoise, p['noiseFloor'] ** 2)
    dof = p['noiseDof']

    out = pd.DataFrame(index=pd.Index(sorted(runs), name='cobra_id'),
                       columns=['gain', 'noise', 'n', 'rejected', 'split'], dtype=float)
    for c, y in runs.items():
        rejected = split = 0
        if len(y) >= 4:
            mssd = np.sum(np.diff(y) ** 2) / (2 * (len(y) - 1))
            z, jump, k = _regimeSplit(y, (dof * fleetNoise + (len(y) - 1) * mssd) / (dof + len(y) - 1))
            if z > p['regime'] and jump > p['regimeJump']:
                y, split = y[k:], k

        noise = fleetNoise
        if len(y) >= 3:
            centre = np.median(y)
            keep = np.abs(y - centre) <= p['reject'] * np.sqrt(fleetNoise)
            rejected, y = int((~keep).sum()), y[keep]
        if len(y) >= 2:
            noise = max((dof * fleetNoise + ((y - y.mean()) ** 2).sum()) / (dof + len(y) - 1),
                        p['noiseFloor'] ** 2)
        if len(y) >= 3:
            mad = (1.4826 * np.median(np.abs(y - np.median(y)))) ** 2
            scale = (dof * fleetNoise + (len(y) - 1) * mad) / (dof + len(y) - 1)
            bound = p['huber'] * np.sqrt(scale)
            gain = np.clip(y, np.median(y) - bound, np.median(y) + bound).mean()
        else:
            gain = y.mean() if len(y) else np.nan
        out.loc[c] = gain, noise, len(y), rejected, split
    return out, float(np.sqrt(fleetNoise))


def _driftRate(ratio, cobraIds, visits):
    """Random-walk step per run of the gains, from their variogram over the runs.

    With the fleet median of each run removed, half the variance of the differences
    between runs `lag` apart is noise + lag * q / 2 for gains that wander by a variance q
    per run on top of their per-run noise; q is read from the shortest and longest lags.
    Visits are in time order and taken as evenly spaced.

    Returns
    -------
    `float`
        sqrt(q), as a gain per run: 0 if no wander is seen, and below about 0.01 not
        distinguished from 0 on a fleet; NaN with fewer than three runs.
    """
    table = pd.DataFrame(dict(ratio=ratio, cobra_id=cobraIds, visit=visits))
    table['ratio'] -= table.groupby('visit').ratio.transform('median')
    runs = table.pivot_table(index='cobra_id', columns='visit', values='ratio').sort_index(axis=1)
    runs = runs.to_numpy(dtype=float)
    k = runs.shape[1] - 1
    if k < 2:
        return np.nan
    gamma = [_robustVariance((runs[:, lag:] - runs[:, :-lag]).ravel() / np.sqrt(2)) for lag in (1, k)]
    return float(np.sqrt(max(2 * (gamma[1] - gamma[0]) / (k - 1), 0.0)))


def _fleetTrend(ratio, visits):
    """Slope of the fleet median gain over the runs, per run; NaN with fewer than two runs.

    Visits are in time order and taken as evenly spaced.
    """
    medians = pd.Series(ratio).groupby(np.asarray(visits)).median().sort_index().to_numpy()
    if len(medians) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(medians)), medians, 1)[0])


def _shiftFrom(gain, n, noise, reference):
    """How the gains measured differ from `reference` gains, over the cobras having both.

    Returns
    -------
    `dict`
        n, median difference, and tau: the spread of the differences beyond what the
        noise of the runs accounts for (0 if none).
    """
    reference = reference.reindex(gain.index)
    both = (n > 0) & np.isfinite(gain) & np.isfinite(reference)
    diff = (gain - reference)[both]
    if both.sum() < 10:
        return dict(n=int(both.sum()), median=np.nan, tau=np.nan)
    excess = _robustVariance(diff) - float(np.median((noise / n)[both]))
    return dict(n=int(both.sum()), median=float(np.median(diff)), tau=float(np.sqrt(max(excess, 0.0))))


def operatingPointOf(meas):
    """The landing depth and push target the rows were taken at.

    Rows of cobras that were not pushed carry no target and are ignored for it.

    Raises
    ------
    `ValueError`
        If the rows are at more than one landing depth or target.
    """
    point = []
    for name in ('dot_landing', 'dot_target'):
        values = np.unique(meas[name].dropna().to_numpy(dtype=float))
        if not len(values) or not np.allclose(values, values[0], atol=1e-6):
            raise ValueError(f'rows are not at one {name}: {values.tolist()}')
        point.append(float(values[0]))
    return tuple(point)


def fitOperatingPoint(meas, prior=None, params=None, minRuns=None, cut=RAMP_RESIDUAL_CUT):
    """Landing and push gain per cobra, measured on runs at the operating point.

    Each gain is the robust mean of the cobra's runs (see `_fitMove`).  The prior is only
    compared with them: the summary says how far the gains moved since the prior and, from
    the runs themselves (`_driftRate`), how fast they wander within the session.

    Parameters
    ----------
    meas : `pandas.DataFrame`
        `cobra_blind_move` rows of the runs, at one landing depth and target.
    prior : `pandas.DataFrame`, optional
        `cobra_blind_move_gain` rows of the calibration the runs were aimed through,
        indexed by cobra_id; its calibrated cobras are compared.
    params : `dict`, optional
        Overrides of `OPERATING_POINT`; an unknown key raises `KeyError`.
    minRuns : `int`, optional
        Overrides OPERATING_POINT minRuns.
    cut : `float`
        `RAMP_RESIDUAL_CUT`.

    Returns
    -------
    (`pandas.DataFrame`, `dict`)
        Indexed by cobra_id: landing_gain, target_gain, n_runs (runs the gain of the move
        with fewer is the mean of, after a change of regime possibly fewer than minRuns),
        landing_scatter, target_scatter (per-run noise, as gains), calibrated, reason (why
        not, else None).  And a summary: the parameters, per move the fleet's
        per-run noise, the drift rate of a cobra, the trend of the fleet median, runs
        rejected and cobras whose earlier runs were dropped at a change of regime, and the
        shifts from the prior.
    """
    unknown = set(params or {}) - set(OPERATING_POINT)
    if unknown:
        raise KeyError(f'unknown operating-point parameters {sorted(unknown)}')
    p = {**OPERATING_POINT, **(params or {})}
    minRuns = p['minRuns'] if minRuns is None else int(minRuns)

    ratios = _moveRatios(meas, p, cut)
    fits, summary = {}, dict(parameters=p, minRuns=minRuns)
    for move in ('landing', 'push'):
        ratio, cobraIds, visits = ratios[move]
        fits[move], noise = _fitMove(ratio, cobraIds, p)
        fit = fits[move]
        summary[move] = dict(noise=noise, drift=_driftRate(ratio, cobraIds, visits),
                             trend=_fleetTrend(ratio, visits),
                             nRejected=int(fit.rejected.sum()), nSplit=int((fit.split > 0).sum()))

    landing, push = fits['landing'], fits['push']
    index = landing.index.union(push.index)
    landing, push = landing.reindex(index), push.reindex(index)
    out = pd.DataFrame(index=index)
    out['landing_gain'] = landing.gain
    out['target_gain'] = push.gain
    out['n_runs'] = np.minimum(landing.n.fillna(0), push.n.fillna(0)).astype(int)
    out['landing_scatter'] = np.sqrt(landing.noise)
    out['target_scatter'] = np.sqrt(push.noise)

    ratio, cobraIds, _ = ratios['landing']
    stalls = pd.Series(ratio < p['stallGain']).groupby(cobraIds).sum().reindex(index).fillna(0)
    few = ((landing.n + landing.split).fillna(0) < minRuns) | ((push.n + push.split).fillna(0) < minRuns)
    lo, hi = dotState.GAIN_BOUNDS
    bounded = out.landing_gain.between(lo, hi) & out.target_gain.between(lo, hi)
    reasons = [('stalled landing', stalls >= 2), (f'fewer than {minRuns} runs', few),
               ('gain outside its bounds', ~bounded)]
    out['reason'] = None
    for reason, mask in reversed(reasons):
        out.loc[mask.to_numpy(), 'reason'] = reason
    out['calibrated'] = out.reason.isna()
    summary['nCalibrated'] = int(out.calibrated.sum())

    if prior is not None:
        calibratedPrior = prior.calibrated.astype('boolean').fillna(False).astype(bool)
        summary['prior'] = {
            move: _shiftFrom(fit.gain, fit.n, fit.noise, prior[col].astype(float).where(calibratedPrior))
            for move, fit, col in (('landing', landing, 'landing_gain'), ('push', push, 'target_gain'))}
    return out, summary


def reachedDepth(meas, phiPushed=None):
    """Depth, in dot fraction, each row reached after its push."""
    phiPushed = meas.phi_pushed.to_numpy(dtype=float) if phiPushed is None else phiPushed
    with np.errstate(invalid='ignore', divide='ignore'):
        return (meas.dot_landing.to_numpy(dtype=float)
                + (phiPushed - meas.phi_landing_target.to_numpy(dtype=float)) / chordOf(meas))


def landedDepth(meas):
    """Depth, in dot fraction, each row reached after its landing."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return (meas.dot_landing.to_numpy(dtype=float)
                + (meas.phi_landed.to_numpy(dtype=float)
                   - meas.phi_landing_target.to_numpy(dtype=float)) / chordOf(meas))


def depthStatistics(error):
    """Median, 16-84% half-width, and fractions within 0.05 and 0.10 of a depth error."""
    error = np.asarray(error, dtype=float)
    error = error[np.isfinite(error)]
    if not len(error):
        return dict(n=0, bias=np.nan, spread=np.nan, within_05=np.nan, within_10=np.nan)
    p16, p50, p84 = np.percentile(error, (16, 50, 84))
    return dict(n=len(error), bias=p50, spread=(p84 - p16) / 2,
                within_05=float(np.mean(np.abs(error) < 0.05)),
                within_10=float(np.mean(np.abs(error) < 0.10)))


def leaveOneOut(meas, cut=RAMP_RESIDUAL_CUT, minRuns=MIN_RUNS):
    """Depth error of each visit aimed from the gains of the others.

    Each held-out visit is scored by `counterfactualErrors`: its own landing and push gains,
    with the moves commanded as the other visits' gains would command them, the landing
    aimed where the visit aimed it.

    A cobra counts only where the other visits calibrate it, so a cobra with exactly
    `minRuns` usable runs is never scored, and fewer than `minRuns` + 1 visits score none.

    Returns
    -------
    `numpy.ndarray` of `float`
        Reached minus wanted depth, one value per usable, calibrated cobra and visit.
    """
    errors = []
    for visit in np.unique(meas.pfs_visit_id):
        gains = fitGains(meas, exclude=visit, cut=cut, minRuns=minRuns)
        held = meas[(meas.pfs_visit_id == visit).to_numpy()]
        _, final = counterfactualErrors(held, gains, aimLanding=landingAimed(held), cut=cut)
        _, _, calibrated = gainsFor(gains, held[usable(held, cut=cut)].cobra_id.to_numpy())
        errors.append(final[calibrated])
    return np.concatenate(errors) if errors else np.array([])


def gainsFor(gains, cobraIds):
    """Landing and push gain of each cobra, 1.0 where `gains` does not calibrate it.

    Parameters
    ----------
    gains : `pandas.DataFrame`
        `cobra_blind_move_gain` rows, indexed by cobra_id.
    cobraIds : `numpy.ndarray` of `int`
        One-based.

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`, `numpy.ndarray`)
        landingGain, targetGain, and the calibrated mask, one value per cobra id.
    """
    g = gains.reindex(np.asarray(cobraIds))
    calibrated = g.calibrated.astype('boolean').fillna(False).to_numpy(dtype=bool)
    return (np.where(calibrated, g.landing_gain.to_numpy(dtype=float), 1.0),
            np.where(calibrated, g.target_gain.to_numpy(dtype=float), 1.0), calibrated)


def landingAimed(meas, tolerance=AIM_TOLERANCE):
    """Per row, whether its visit commanded some landing to another depth than dot_landing.

    Rows from before the commanded angles were recorded never are.
    """
    command = meas.phi_landing_command if 'phi_landing_command' in meas else pd.Series(np.nan, index=meas.index)
    with np.errstate(invalid='ignore', divide='ignore'):
        moved = (np.abs(command - meas.phi_landing_target) / np.abs(chordOf(meas))).to_numpy() > tolerance
    return pd.Series(moved, index=meas.index).groupby(meas.pfs_visit_id.to_numpy()).transform('any').to_numpy()


def checkAimedBy(meas, gains, calibId, approach=dotGeometry.RAMP_APPROACH_FRACTION):
    """Rows not aimed through calibration `calibId`, whose gains are `gains`, as a mask.

    A row records the calibration that aimed it, set on the cobras that calibration
    calibrates, and where its visit aimed the landing, the landing command must be the one
    `landingCommands` gives.  Rows from before those columns are checked by recomputing
    their command: a cobra without calibrated gains is aimed with gain 1.0, so its command
    is its wanted depth.  A row that was not pushed has no chord to check its landing with.
    """
    g1, g2, calibrated = gainsFor(gains, meas.cobra_id.to_numpy())
    legacy = _legacy(meas)
    recordedBy = (meas.blind_move_calib_id if 'blind_move_calib_id' in meas
                  else pd.Series(np.nan, index=meas.index)).astype(float).to_numpy()
    notAimed = calibrated & (recordedBy != calibId)

    if 'phi_landing_command' in meas:
        chord = chordOf(meas)
        landing = meas.dot_landing.to_numpy(dtype=float)
        expected = (meas.phi_landing_target.to_numpy(dtype=float)
                    + (landingCommands(landing, approach, g1, calibrated) - landing) * chord)
        with np.errstate(invalid='ignore', divide='ignore'):
            wrong = np.abs(meas.phi_landing_command.to_numpy(dtype=float) - expected) / np.abs(chord)
        notAimed |= landingAimed(meas) & calibrated & np.isfinite(chord) & ~(wrong <= AIM_TOLERANCE)

    expected, _ = aimedCommand(meas.dot_target.to_numpy(), meas.dot_landing.to_numpy(),
                               meas.phi_start.to_numpy(), meas.phi_landing_target.to_numpy(),
                               chordOf(meas), g1, g2)
    recorded = meas.dot_command.to_numpy(dtype=float)
    legacyNotAimed = np.isfinite(recorded) & ~(np.abs(recorded - expected) <= AIM_TOLERANCE)
    return np.where(legacy, legacyNotAimed, notAimed)


def counterfactualErrors(meas, gains, *, aimLanding, approach=dotGeometry.RAMP_APPROACH_FRACTION,
                         cut=RAMP_RESIDUAL_CUT):
    """Landing and final depth errors each usable row would have had with `gains`.

    Each row keeps its own measured start and its own response -- its landing and push
    gains in that run -- and the moves are commanded as `gains` would command them: the
    landing to `landingCommands`, or to dot_landing where aimLanding (a `bool`, or one per
    row of `meas`) is False, and the push sized from the landing the gains predict and
    capped at `dotGeometry.DEPTH_CAP` as `dotMove.blindMoveToDots` caps it.  Comparing
    two calibrations on the same rows so removes the scatter of the runs from the
    comparison.

    Returns
    -------
    (`numpy.ndarray`, `numpy.ndarray`)
        Landing depth less dot_landing, and final depth less dot_target, in dot fraction,
        of the rows `usable` keeps.
    """
    ok = usable(meas, cut=cut)
    aimLanding = np.broadcast_to(np.asarray(aimLanding, dtype=bool), ok.shape)[ok]
    use = meas[ok]
    travels = moveTravels(use)
    with np.errstate(invalid='ignore', divide='ignore'):
        ownLanding = (travels.y1 / travels.x1).to_numpy()
        ownPush = (travels.y2 / travels.x2).to_numpy()
    chord = chordOf(use)
    landing = use.dot_landing.to_numpy(dtype=float)
    target = use.dot_target.to_numpy(dtype=float)
    phiTarget = use.phi_landing_target.to_numpy(dtype=float)
    start = use.phi_start.to_numpy(dtype=float)

    g1, g2, calibrated = gainsFor(gains, use.cobra_id.to_numpy())
    g1, g2 = boundedGains(g1), boundedGains(g2)
    command = np.where(aimLanding, landingCommands(landing, approach, g1, calibrated), landing)
    phiCommand = phiTarget + (command - landing) * chord
    landed = start + ownLanding * (phiCommand - start)
    predicted = start + g1 * (phiCommand - start)
    shallow, deep = dotGeometry.DEPTH_CAP
    with np.errstate(invalid='ignore', divide='ignore'):
        depth = (phiTarget + (target - landing) * chord - predicted) / g2 / chord
        estimate = (predicted - phiTarget) / chord + landing
        # At gain 1 the landing ends at its command.
        room = np.where(depth > 0, deep - np.maximum(estimate, command),
                        shallow - np.minimum(estimate, command))
        depth = np.where(depth > 0, np.clip(depth, None, np.maximum(room, 0.0)),
                         np.clip(depth, np.minimum(room, 0.0), None))
    reached = landed + ownPush * depth * chord
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((landed - phiTarget) / chord,
                (reached - phiTarget) / chord - (target - landing))


def _rms(errors, clip=None):
    """Root mean square of the finite `errors`; with `clip`, of the errors less their median
    and clipped at +-`clip`."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if clip is not None and len(errors):
        errors = np.clip(errors - np.median(errors), -clip, clip)
    return float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan


def validate(meas, gains, calibId, criteria=None, cut=RAMP_RESIDUAL_CUT, reference=None,
             alsoUnaimed=False):
    """Judge a calibration on visits aimed through it.

    Parameters
    ----------
    meas : `pandas.DataFrame`
        `cobra_blind_move` rows of the aimed visits.
    gains : `pandas.DataFrame`
        `cobra_blind_move_gain` rows of the calibration, indexed by cobra_id.
    calibId : `int`
        The calibration's id, which the rows must record as their aim.
    criteria : `dict`, optional
        Overrides of `VALIDATION_CRITERIA`; an unknown key raises `KeyError`.
    cut : `float`
    reference : `pandas.DataFrame`, optional
        Gains of the calibration it would replace: the calibration must not do worse on
        the same rows, by `counterfactualErrors`.
    alsoUnaimed : `bool`
        With a reference, compare the two also with every landing unaimed.

    Returns
    -------
    (`bool`, `dict`, `list` of `str`)
        Whether it passed, the statistics of the `blind_move_calib_validation` row, and
        the reasons it failed.  The statistics also hold 'landing' (landing depth less
        dot_landing of the calibrated cobras, with 'short' over all cobras and
        'shortCalibrated'), 'landingSlow' (the same for the cobras calibrated below
        `SLOW_LANDING_GAIN`, reported, not judged), and with a reference, 'regression': the
        counterfactual rms of the calibration and the reference, of the errors less their
        median and clipped, and 'Unclipped', of the errors as they are; with alsoUnaimed,
        'regressionUnaimed' the same with every landing unaimed.
    """
    unknown = set(criteria or {}) - set(VALIDATION_CRITERIA)
    if unknown:
        raise KeyError(f'unknown validation criteria {sorted(unknown)}')
    criteria = {**VALIDATION_CRITERIA, **(criteria or {})}
    failures = []

    notAimed = checkAimedBy(meas, gains, calibId)
    if notAimed.any():
        visits = sorted(set(meas.pfs_visit_id.to_numpy()[notAimed].tolist()))
        failures.append(f'{int(notAimed.sum())} cobras of visits {visits} not aimed by this '
                        f'calibration')
    aimed = landingAimed(meas)
    if gains.attrs.get('calibration', {}).get('model') == OPERATING_POINT_MODEL and not aimed.all():
        visits = sorted(set(meas.pfs_visit_id.to_numpy()[~aimed].tolist()))
        failures.append(f'the landing of visits {visits} was not aimed')

    ok = usable(meas, cut=cut)
    error = reachedDepth(meas[ok]) - meas.dot_target.to_numpy()[ok]
    stats = depthStatistics(error)
    stats['coverage'] = float(np.isfinite(error).sum() / len(meas)) if len(meas) else np.nan
    landingError = landedDepth(meas[ok]) - meas.dot_landing.to_numpy()[ok]

    if not stats['n']:
        failures.append('no cobra measured')
    else:
        if not abs(stats['bias']) <= criteria['maxBias']:
            failures.append(f'bias {stats["bias"]:+.3f} beyond {criteria["maxBias"]}')
        if not stats['spread'] <= criteria['maxSpread']:
            failures.append(f'spread {stats["spread"]:.3f} above {criteria["maxSpread"]}')
        if not stats['within_05'] >= criteria['minWithin05']:
            failures.append(f'{stats["within_05"]:.0%} within 0.05, below '
                            f'{criteria["minWithin05"]:.0%}')
        if not stats['coverage'] >= criteria['minCoverage']:
            failures.append(f'coverage {stats["coverage"]:.0%} below {criteria["minCoverage"]:.0%}')

    g1, _, calibrated = gainsFor(gains, meas.cobra_id.to_numpy()[ok])
    aimedRow = aimed[ok]
    measured = aimedRow & np.isfinite(landingError)
    stats['landing'] = landing = depthStatistics(landingError[calibrated & aimedRow])
    stats['landingSlow'] = depthStatistics(landingError[calibrated & aimedRow & (g1 < SLOW_LANDING_GAIN)])
    if landing['n']:
        landing['short'] = float(np.mean(landingError[measured] < -0.08))
        landing['shortCalibrated'] = float(np.mean(landingError[measured & calibrated] < -0.08))
        if not abs(landing['bias']) <= criteria['maxLandingBias']:
            failures.append(f'landing bias {landing["bias"]:+.3f} beyond {criteria["maxLandingBias"]}')
        if not landing['spread'] <= criteria['maxLandingSpread']:
            failures.append(f'landing spread {landing["spread"]:.3f} above '
                            f'{criteria["maxLandingSpread"]}')
        if not landing['within_05'] >= criteria['minLandingWithin05']:
            failures.append(f'{landing["within_05"]:.0%} of landings within 0.05, below '
                            f'{criteria["minLandingWithin05"]:.0%}')
        if not landing['short'] <= criteria['maxShortLandings']:
            failures.append(f'{landing["short"]:.1%} of landings short by more than 0.08, above '
                            f'{criteria["maxShortLandings"]:.1%}')

    comparisons = {} if reference is None else dict(regression=aimed)
    if reference is not None and alsoUnaimed and aimed.any():
        comparisons['regressionUnaimed'] = np.zeros(len(meas), dtype=bool)
    for key, aim in comparisons.items():
        rms = {}
        for name, g in (('calibration', gains), ('reference', reference)):
            errors = counterfactualErrors(meas, g, aimLanding=aim, cut=cut)
            rms[name] = {}
            for move, e in zip(('landing', 'final'), errors):
                rms[name][move] = _rms(e, clip=REGRESSION_CLIP)
                rms[name][f'{move}Unclipped'] = _rms(e)
        stats[key] = rms
        unaimed = ' with the landing unaimed' if key == 'regressionUnaimed' else ''
        for move in ('landing', 'final') if aim.any() else ('final',):
            if not rms['calibration'][move] <= rms['reference'][move] + criteria['maxRegression']:
                failures.append(f'counterfactual {move} rms{unaimed} {rms["calibration"][move]:.3f} '
                                f'worse than the reference {rms["reference"][move]:.3f}')

    return not failures, stats, failures


def loadMeasurements(db, visits):
    """`cobra_blind_move` rows of `visits`, as a `pandas.DataFrame`."""
    visits = [int(v) for v in visits]
    meas = db.query_dataframe(
        f'SELECT {", ".join(MEASUREMENT_COLUMNS)} FROM cobra_blind_move '
        f'WHERE pfs_visit_id IN ({", ".join(str(v) for v in visits)}) '
        f'ORDER BY pfs_visit_id, cobra_id')
    missing = sorted(set(visits) - set(meas.pfs_visit_id.astype(int).tolist()))
    if missing:
        raise ValueError(f'no cobra_blind_move rows for visits {missing}')
    return meas


def loadGains(db, calibId):
    """`cobra_blind_move_gain` rows of a calibration, indexed by cobra_id.

    attrs['calibration'] holds the calibration's model, the prior it was aimed through and,
    for an operating-point calibration, the landing and target depth it was measured at;
    empty if the table has no such columns.
    """
    gains = db.query_dataframe(
        'SELECT cobra_id, landing_gain, target_gain, n_runs, landing_scatter, target_scatter, '
        f'calibrated FROM cobra_blind_move_gain WHERE blind_move_calib_id = {int(calibId)}')
    if gains.empty:
        raise ValueError(f'no blind-move calibration {calibId}')
    gains = gains.set_index('cobra_id')
    try:
        info = db.query_dataframe(
            'SELECT model, prior_calib_id, dot_landing, dot_target FROM blind_move_calib '
            f'WHERE blind_move_calib_id = {int(calibId)}')
    except exc.ProgrammingError:
        info = pd.DataFrame()
    gains.attrs['calibration'] = {k: _nullable(v) for k, v in info.iloc[0].items()} if len(info) else {}
    return gains


def hoursAfterCalibration(db, calibId, visits):
    """Hours from the last run of calibration `calibId` to the first of `visits`; NaN if
    either has no visit time."""
    age = db.query_dataframe(
        'SELECT extract(epoch FROM (SELECT min(issued_at) FROM pfs_visit '
        'WHERE pfs_visit_id = ANY(:visits)) - (SELECT max(v.issued_at) FROM pfs_visit v, '
        'blind_move_calib c WHERE c.blind_move_calib_id = :calib AND '
        'v.pfs_visit_id = ANY(c.visits))) AS seconds',
        params=dict(visits=[int(v) for v in visits], calib=int(calibId)))
    seconds = age.seconds.iloc[0] if len(age) else None
    return np.nan if seconds is None or pd.isna(seconds) else float(seconds) / 3600


def defaultCalibId(db):
    """The default calibration, that of the newest validation row; None if there is none."""
    rows = db.query_dataframe(
        'SELECT blind_move_calib_id FROM blind_move_calib_validation '
        'ORDER BY validated_at DESC, blind_move_calib_validation_id DESC LIMIT 1')
    return None if rows.empty else int(rows.blind_move_calib_id.iloc[0])


def lastValidation(db, calibId=None, notCalibId=None):
    """The newest validation row of `calibId`, or of any calibration but `notCalibId`.

    Returns
    -------
    `dict` or None
        blind_move_calib_id and visits of the row.
    """
    where = (f'WHERE blind_move_calib_id = {int(calibId)}' if calibId is not None
             else f'WHERE blind_move_calib_id <> {int(notCalibId)}' if notCalibId is not None
             else '')
    rows = db.query_dataframe(
        f'SELECT blind_move_calib_id, visits FROM blind_move_calib_validation {where} '
        'ORDER BY validated_at DESC, blind_move_calib_validation_id DESC LIMIT 1')
    if rows.empty:
        return None
    return dict(blind_move_calib_id=int(rows.blind_move_calib_id.iloc[0]),
                visits=[int(v) for v in rows.visits.iloc[0]])


def _jsonSafe(value):
    """`value` with numpy scalars made Python ones and non-finite numbers None, recursively."""
    if isinstance(value, dict):
        return {str(k): _jsonSafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonSafe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _nullable(value):
    """A value as SQL takes it: NaN and pandas NA become None, numpy scalars Python ones."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return value.item() if isinstance(value, np.generic) else value


def insertCalibration(db, visits, gains, looStats, cut=RAMP_RESIDUAL_CUT, minRuns=MIN_RUNS,
                      model=MODEL, modelVersion=MODEL_VERSION, priorCalibId=None,
                      dotLanding=None, dotTarget=None, parameters=None):
    """Write a calibration and its gains in one transaction.

    Parameters
    ----------
    looStats : `dict` or None
        `depthStatistics` of the leave-one-out error, None where the model has none.
    parameters : `dict`, optional
        What the fit was run with and found, stored as JSON; non-finite numbers as null.

    Returns
    -------
    `int`
        The new blind_move_calib_id.
    """
    loo = looStats or {}
    with db.connection() as conn:
        calibId = conn.execute(text(
            'INSERT INTO blind_move_calib (created_at, visits, ramp_residual_cut, min_runs, '
            'model, model_version, n_calibrated, loo_bias, loo_spread, loo_within_05, '
            'prior_calib_id, dot_landing, dot_target, parameters) '
            'VALUES (now(), :visits, :cut, :minRuns, :model, :version, :nCalibrated, '
            ':bias, :spread, :within05, :prior, :dotLanding, :dotTarget, :parameters) '
            'RETURNING blind_move_calib_id'),
            dict(visits=[int(v) for v in visits], cut=float(cut), minRuns=int(minRuns),
                 model=model, version=int(modelVersion), nCalibrated=int(gains.calibrated.sum()),
                 bias=_nullable(loo.get('bias')), spread=_nullable(loo.get('spread')),
                 within05=_nullable(loo.get('within_05')), prior=_nullable(priorCalibId),
                 dotLanding=_nullable(dotLanding), dotTarget=_nullable(dotTarget),
                 parameters=None if parameters is None else json.dumps(_jsonSafe(parameters),
                                                                        allow_nan=False))
        ).scalar_one()
        rows = [dict(calib=calibId, cobra=int(cobraId),
                     **{name: _nullable(row[name]) for name in
                        ('landing_gain', 'target_gain', 'n_runs', 'landing_scatter',
                         'target_scatter')},
                     calibrated=bool(row['calibrated']), reason=_nullable(row.get('reason')))
                for cobraId, row in gains.iterrows()]
        conn.execute(text(
            'INSERT INTO cobra_blind_move_gain (blind_move_calib_id, cobra_id, landing_gain, '
            'target_gain, n_runs, landing_scatter, target_scatter, calibrated, reason) VALUES '
            '(:calib, :cobra, :landing_gain, :target_gain, :n_runs, :landing_scatter, '
            ':target_scatter, :calibrated, :reason)'), rows)
    return int(calibId)


def insertValidation(db, calibId, visits, stats, reason):
    """Write a validation row, which makes `calibId` the default."""
    with db.connection() as conn:
        conn.execute(text(
            'INSERT INTO blind_move_calib_validation (blind_move_calib_id, validated_at, visits, '
            'n_cobras, bias, spread, within_05, within_10, coverage, reason) VALUES '
            '(:calib, now(), :visits, :n, :bias, :spread, :within05, :within10, :coverage, '
            ':reason)'),
            dict(calib=int(calibId), visits=[int(v) for v in visits], n=int(stats['n']),
                 bias=_nullable(stats['bias']), spread=_nullable(stats['spread']),
                 within05=_nullable(stats['within_05']), within10=_nullable(stats['within_10']),
                 coverage=_nullable(stats['coverage']), reason=reason))
