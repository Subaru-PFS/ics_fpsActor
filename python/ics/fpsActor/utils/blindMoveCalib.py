"""Fit, validate and select the per-cobra calibration of the two blind moves.

The measurements are the rows of opdb `cobra_blind_move`, one per cobra and turned-dot
visit; everything here is computed from their columns alone, including the dot chord,
so a calibration can be refitted or checked without the run directories or the geometry.

A calibration is written once (`blind_move_calib`, `cobra_blind_move_gain`) and becomes
the default when it is judged good on aimed runs (`blind_move_calib_validation`): the
default is the calibration of the newest validation row.  Nothing is ever updated.
"""
import numpy as np
import pandas as pd
from sqlalchemy import text

MODEL = 'gain-lsq'
"""Gain-only model, least squares through the origin."""

MODEL_VERSION = 1

RAMP_RESIDUAL_CUT = 0.100
"""Largest distance from target before the landing, mm, a move is fitted with.  Past it
the cobra did not start the landing where the ramp put it."""

MIN_RUNS = 3
"""Passing runs a cobra needs before its gains are used."""

GAIN_BOUNDS = (0.5, 2.0)
"""Gains outside this measured something other than the motor."""

AIM_TOLERANCE = 1e-3
"""Largest difference, in dot fraction, between a recorded and a recomputed command for a
visit to count as aimed by a calibration."""

VALIDATION_CRITERIA = dict(maxBias=0.02, maxSpread=0.05, minWithin05=0.75, minCoverage=0.90)
"""Defaults of the validation: |median error|, 16-84% half-width, fraction within 0.05 of
the depth wanted, and fraction of the dot cobras measured, over all aimed visits."""

MEASUREMENT_COLUMNS = ['pfs_visit_id', 'cobra_id', 'dot_landing', 'dot_target', 'dot_command',
                       'phi_landing_target', 'phi_push_target', 'phi_start', 'phi_landed',
                       'phi_pushed', 'ramp_residual']


def chordOf(meas):
    """Signed phi per unit dot fraction, radians, of each measurement row.

    NaN where the push was commanded no deeper than the landing, which defines no chord.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        chord = ((meas.phi_push_target - meas.phi_landing_target)
                 / (meas.dot_command - meas.dot_landing)).to_numpy(dtype=float)
    return np.where(np.isfinite(chord) & (chord != 0), chord, np.nan)


def usable(meas, cut=RAMP_RESIDUAL_CUT):
    """Rows whose two moves were measured and started where the ramp put them."""
    measured = meas[['phi_start', 'phi_landed', 'phi_pushed', 'phi_landing_target',
                     'phi_push_target']].notna().all(axis=1)
    return (measured & (meas.ramp_residual <= cut)).to_numpy()


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

    Each gain is the measured phi change over the commanded one.  The landing is
    commanded from the measured phi_start to phi_landing_target.  The push is commanded
    from phi_landing_target, which is where it was sized from: the tracker's estimate
    after an unmeasured landing, with tracker gain 1.0 and the landing at phi scale 1.0.
    Its measured change starts from phi_landed, so the two gains are independent.  A push
    clipped at `dotGeometry.MAX_BLIND_STEPS` reads as a low push gain.

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
    x1 = use.phi_landing_target - use.phi_start
    y1 = use.phi_landed - use.phi_start
    x2 = use.phi_push_target - use.phi_landing_target
    y2 = use.phi_pushed - use.phi_landed
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
                         & out.landing_gain.between(*GAIN_BOUNDS)
                         & out.target_gain.between(*GAIN_BOUNDS))
    return out


def reachedDepth(meas, phiPushed=None):
    """Depth, in dot fraction, each row reached after its push."""
    phiPushed = meas.phi_pushed.to_numpy(dtype=float) if phiPushed is None else phiPushed
    with np.errstate(invalid='ignore', divide='ignore'):
        return (meas.dot_landing.to_numpy(dtype=float)
                + (phiPushed - meas.phi_landing_target.to_numpy(dtype=float)) / chordOf(meas))


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

    Where the held-out visit would have gone is its own measured landing plus its own push
    gain times the push the other visits' gains would have commanded, so nothing is assumed
    beyond the push being linear in its size.

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
        held = meas[usable(meas, cut=cut) & (meas.pfs_visit_id == visit).to_numpy()]
        held = held[held.cobra_id.isin(gains.index[gains.calibrated]).to_numpy()]
        g = gains.loc[held.cobra_id]
        chord = chordOf(held)
        with np.errstate(invalid='ignore', divide='ignore'):
            ownTargetGain = ((held.phi_pushed - held.phi_landed)
                             / (held.phi_push_target - held.phi_landing_target)).to_numpy()
        command, _ = aimedCommand(held.dot_target.to_numpy(), held.dot_landing.to_numpy(),
                                  held.phi_start.to_numpy(), held.phi_landing_target.to_numpy(),
                                  chord, g.landing_gain.to_numpy(), g.target_gain.to_numpy())
        phiReached = (held.phi_landed.to_numpy()
                      + ownTargetGain * (command - held.dot_landing.to_numpy()) * chord)
        errors.append(reachedDepth(held, phiPushed=phiReached) - held.dot_target.to_numpy())
    return np.concatenate(errors) if errors else np.array([])


def checkAimedBy(meas, gains):
    """Rows whose recorded command is not the one `gains` gives, as a boolean mask.

    A cobra without calibrated gains is aimed with gain 1.0, so its command is its wanted
    depth.
    """
    g = gains.reindex(meas.cobra_id)
    calibrated = g.calibrated.astype('boolean').fillna(False).to_numpy(dtype=bool)
    g1 = np.where(calibrated, g.landing_gain.to_numpy(dtype=float), 1.0)
    g2 = np.where(calibrated, g.target_gain.to_numpy(dtype=float), 1.0)
    expected, _ = aimedCommand(meas.dot_target.to_numpy(), meas.dot_landing.to_numpy(),
                               meas.phi_start.to_numpy(), meas.phi_landing_target.to_numpy(),
                               chordOf(meas), g1, g2)
    recorded = meas.dot_command.to_numpy(dtype=float)
    return np.isfinite(recorded) & ~(np.abs(recorded - expected) <= AIM_TOLERANCE)


def validate(meas, gains, criteria=None, cut=RAMP_RESIDUAL_CUT):
    """Judge a calibration on visits aimed through it.

    Parameters
    ----------
    meas : `pandas.DataFrame`
        `cobra_blind_move` rows of the aimed visits.
    gains : `pandas.DataFrame`
        `cobra_blind_move_gain` rows of the calibration, indexed by cobra_id.
    criteria : `dict`, optional
        Overrides of `VALIDATION_CRITERIA`; an unknown key raises `KeyError`.
    cut : `float`

    Returns
    -------
    (`bool`, `dict`, `list` of `str`)
        Whether it passed, the statistics of the `blind_move_calib_validation` row, and
        the reasons it failed.
    """
    unknown = set(criteria or {}) - set(VALIDATION_CRITERIA)
    if unknown:
        raise KeyError(f'unknown validation criteria {sorted(unknown)}')
    criteria = {**VALIDATION_CRITERIA, **(criteria or {})}
    failures = []

    notAimed = checkAimedBy(meas, gains)
    if notAimed.any():
        visits = sorted(set(meas.pfs_visit_id.to_numpy()[notAimed].tolist()))
        failures.append(f'{int(notAimed.sum())} cobras of visits {visits} not aimed by this '
                        f'calibration')

    ok = usable(meas, cut=cut)
    error = reachedDepth(meas[ok]) - meas.dot_target.to_numpy()[ok]
    stats = depthStatistics(error)
    stats['coverage'] = float(np.isfinite(error).sum() / len(meas)) if len(meas) else np.nan

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
    """`cobra_blind_move_gain` rows of a calibration, indexed by cobra_id."""
    gains = db.query_dataframe(
        'SELECT cobra_id, landing_gain, target_gain, n_runs, landing_scatter, target_scatter, '
        f'calibrated FROM cobra_blind_move_gain WHERE blind_move_calib_id = {int(calibId)}')
    if gains.empty:
        raise ValueError(f'no blind-move calibration {calibId}')
    return gains.set_index('cobra_id')


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


def _nullable(value):
    """A value as SQL takes it: NaN and pandas NA become None, numpy scalars Python ones."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return value.item() if isinstance(value, np.generic) else value


def insertCalibration(db, visits, gains, looStats, cut=RAMP_RESIDUAL_CUT, minRuns=MIN_RUNS):
    """Write a calibration and its gains in one transaction.

    Returns
    -------
    `int`
        The new blind_move_calib_id.
    """
    with db.connection() as conn:
        calibId = conn.execute(text(
            'INSERT INTO blind_move_calib (created_at, visits, ramp_residual_cut, min_runs, '
            'model, model_version, n_calibrated, loo_bias, loo_spread, loo_within_05) '
            'VALUES (now(), :visits, :cut, :minRuns, :model, :version, :nCalibrated, '
            ':bias, :spread, :within05) RETURNING blind_move_calib_id'),
            dict(visits=[int(v) for v in visits], cut=float(cut), minRuns=int(minRuns),
                 model=MODEL, version=MODEL_VERSION, nCalibrated=int(gains.calibrated.sum()),
                 bias=_nullable(looStats['bias']), spread=_nullable(looStats['spread']),
                 within05=_nullable(looStats['within_05']))).scalar_one()
        rows = [dict(calib=calibId, cobra=int(cobraId),
                     **{name: _nullable(row[name]) for name in
                        ('landing_gain', 'target_gain', 'n_runs', 'landing_scatter',
                         'target_scatter')},
                     calibrated=bool(row['calibrated']))
                for cobraId, row in gains.iterrows()]
        conn.execute(text(
            'INSERT INTO cobra_blind_move_gain (blind_move_calib_id, cobra_id, landing_gain, '
            'target_gain, n_runs, landing_scatter, target_scatter, calibrated) VALUES '
            '(:calib, :cobra, :landing_gain, :target_gain, :n_runs, :landing_scatter, '
            ':target_scatter, :calibrated)'), rows)
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
