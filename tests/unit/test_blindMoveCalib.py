"""Tests for blindMoveCalib and the aimed push, on synthetic rows and a linear fake map.

The aim must return the wanted depth for unit gains, the fit must recover the gains the
rows were built with, and the push must be sized to the aimed command.
"""
import types

import numpy as np
import pandas as pd
import pytest

from ics.cobraCharmer import func
from ics.fpsActor.utils import blindMoveCalib, dotGeometry, dotMove, dotState

CHORD = 0.30            # rad per unit dot fraction
LANDING = 0.1


def rows(nCobras, g1, g2, wanted, visit=1, phiStart=1.0):
    """cobra_blind_move rows of cobras that respond with exactly `g1` and `g2`."""
    phiLT = 1.2
    phiPT = phiLT + (wanted - LANDING) * CHORD
    phiLanded = phiStart + g1 * (phiLT - phiStart)
    phiPushed = phiLanded + g2 * (phiPT - phiLT)
    return pd.DataFrame(dict(pfs_visit_id=visit, cobra_id=np.arange(1, nCobras + 1),
                             dot_landing=LANDING, dot_target=wanted, dot_command=wanted,
                             phi_landing_target=phiLT, phi_push_target=phiPT,
                             phi_start=phiStart, phi_landed=phiLanded, phi_pushed=phiPushed,
                             ramp_residual=0.01))


def test_unitGainsGiveTheWantedDepth():
    command, aimed = blindMoveCalib.aimedCommand(0.5, LANDING, np.array([1.0, np.nan]), 1.2,
                                                 CHORD, 1.0, 1.0)
    assert np.allclose(command, 0.5) and aimed.tolist() == [True, False]


def test_fitRecoversTheGains():
    meas = pd.concat([rows(3, 1.10, 0.95, w, visit=v) for v, w in ((1, 0.3), (2, 0.5), (3, 0.7))])
    gains = blindMoveCalib.fitGains(meas)
    assert np.allclose(gains.landing_gain, 1.10) and np.allclose(gains.target_gain, 0.95)
    assert gains.calibrated.all() and (gains.n_runs == 3).all()


@pytest.mark.parametrize('g1, g2, calibrated', [(0.3, 2.8, True), (1.0, 4.5, False), (0.2, 1.0, False)])
def test_fitAcceptsTheGainsTheTrackerBounds(g1, g2, calibrated):
    meas = pd.concat([rows(1, g1, g2, w, visit=v) for v, w in ((1, 0.3), (2, 0.5), (3, 0.7))])
    assert blindMoveCalib.fitGains(meas).calibrated.tolist() == [calibrated]


def test_aimedCommandReachesTheWantedDepth():
    g1, g2, wanted, phiStart, phiLT = 1.10, 0.95, 0.5, 1.0, 1.2
    command, _ = blindMoveCalib.aimedCommand(wanted, LANDING, phiStart, phiLT, CHORD, g1, g2)
    landed = phiStart + g1 * (phiLT - phiStart)
    reached = landed + g2 * (command - LANDING) * CHORD        # push sized from phiLT
    assert reached == pytest.approx(phiLT + (wanted - LANDING) * CHORD)


def test_gainsForUncalibratedIsOne():
    gains = pd.DataFrame(dict(landing_gain=[1.1, 1.3], target_gain=[0.9, 1.4],
                              calibrated=[True, False]), index=pd.Index([2, 3], name='cobra_id'))
    g1, g2, calibrated = blindMoveCalib.gainsFor(gains, np.array([1, 2, 3]))
    assert g1.tolist() == [1.0, 1.1, 1.0] and g2.tolist() == [1.0, 0.9, 1.0]
    assert calibrated.tolist() == [False, True, False]


def fakeCc(nCobras):
    return types.SimpleNamespace(
        allCobras=np.array([func.Cobra(1, 1 + i) for i in range(nCobras)]),
        cobraInfo=dict(detected=np.ones(nCobras, bool)), calibModel=None,
        pfi=types.SimpleNamespace(moveSteps=lambda cobras, t, p, phiFast=True: None))


def patchGeometry(monkeypatch, nCobras):
    ones = np.ones(nCobras)
    monkeypatch.setattr(dotGeometry, 'computeDotAngles',
                        lambda cc, dotThetaOffset=0.0: (None, np.zeros(nCobras), None, None, None,
                                                        ones, ones))
    monkeypatch.setattr(dotState, 'stepsToTarget', lambda *a, **k: 100)


def tracker(nCobras, calibrated=None):
    return types.SimpleNamespace(phi=np.zeros(nCobras), phiAtGainOne=np.zeros(nCobras),
                                 gain=np.ones(nCobras),
                                 calibrated=(np.zeros(nCobras, bool) if calibrated is None
                                             else np.asarray(calibrated)),
                                 applied=lambda m, s: None, lastPush=None)


def test_pushRecordsWhereItWasSizedFrom(monkeypatch):
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)
    monkeypatch.setattr(dotState, 'mapSpan', lambda model, c, steps, start: start + 0.001 * steps)
    t = tracker(nCobras, calibrated=[False, True, False, False, False, False])
    t.phi[:] = 0.2
    informs = []
    cmd = types.SimpleNamespace(inform=informs.append, warn=informs.append)
    dotMove.blindMoveToDots(fakeCc(nCobras), t, dotIdx, cmd=cmd, targetFraction=0.5,
                            requireHidden=False)
    assert np.allclose(t.lastPush.dotTarget, 0.5)
    assert np.allclose(t.lastPush.phiPushStart, 0.2) and np.allclose(t.lastPush.phiPushCommand, 0.3)
    assert t.lastPush.calibrated.tolist() == [True, False]
    assert any('1/2 cobras aimed' in text for text in informs)


def test_fluxStepRecordsNoPush(monkeypatch):
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)
    t = tracker(nCobras)
    dotMove.blindMoveToDots(fakeCc(nCobras), t, dotIdx, deltaFraction=0.05, requireHidden=False)
    assert t.lastPush is None


def test_defaultDepthIsTheDotCentre(monkeypatch):
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)
    monkeypatch.setattr(dotState, 'mapSpan', lambda model, c, steps, start: start)
    t = tracker(nCobras)
    dotMove.blindMoveToDots(fakeCc(nCobras), t, dotIdx, requireHidden=False)
    assert dotGeometry.BLIND_TARGET_FRACTION == 0.5
    assert np.allclose(t.lastPush.dotTarget, 0.5)


# ── commanded travels, legacy rows ────────────────────────────────────────────

def test_legacyRowsMeasureAgainstTheTargets():
    meas = rows(3, 1.10, 0.95, 0.5)
    travels = blindMoveCalib.moveTravels(meas)
    assert np.allclose(travels.x1, meas.phi_landing_target - meas.phi_start)
    assert np.allclose(travels.x2, meas.phi_push_target - meas.phi_landing_target)


def test_recordedCommandsReplaceTheTargets():
    meas = rows(2, 1.0, 1.0, 0.5).assign(phi_landing_command=[1.3, np.nan], phi_push_start=1.25,
                                         phi_push_command=1.4)
    travels = blindMoveCalib.moveTravels(meas)
    # A row recording its push but not its landing command is not legacy: no landing.
    assert travels.x1[0] == pytest.approx(0.3) and np.isnan(travels.x1[1])
    assert np.allclose(travels.x2, 0.15)


def test_rowsAimedByAnotherCalibrationAreNotAimed():
    gains = pd.DataFrame(dict(landing_gain=[1.1, 1.3], target_gain=[0.9, 1.4],
                              calibrated=[True, False]), index=pd.Index([1, 2], name='cobra_id'))
    meas = rows(2, 1.0, 1.0, 0.5).assign(phi_push_start=1.2, blind_move_calib_id=pd.array([7, pd.NA]))
    assert blindMoveCalib.checkAimedBy(meas, gains, 7).tolist() == [False, False]
    assert blindMoveCalib.checkAimedBy(meas, gains, 8).tolist() == [True, False]


# ── operating-point fit ───────────────────────────────────────────────────────

def opRows(trueG1, trueG2, nRuns, noise=0.04, seed=0, drop=None):
    """Runs at one operating point of cobras with gains trueG1, trueG2 and per-run noise."""
    rng = np.random.default_rng(seed)
    out = []
    for run in range(nRuns):
        r = rows(len(trueG1), 1.0, 1.0, 0.5, visit=100 + run)
        x1 = r.phi_landing_target - r.phi_start
        x2 = r.phi_push_target - r.phi_landing_target
        g1 = np.asarray(trueG1) * (1 + noise * rng.standard_normal(len(r)))
        g2 = np.asarray(trueG2) * (1 + noise * rng.standard_normal(len(r)))
        r['phi_landed'] = r.phi_start + g1 * x1
        r['phi_pushed'] = r.phi_landed + g2 * x2
        out.append(r)
    return pd.concat(out, ignore_index=True)


def priorOf(g1, g2, scatter=0.04, nRuns=9):
    n = len(g1)
    return pd.DataFrame(dict(landing_gain=g1, target_gain=g2, n_runs=nRuns,
                             landing_scatter=scatter, target_scatter=scatter, calibrated=True),
                        index=pd.Index(np.arange(1, n + 1), name='cobra_id'))


def test_gainsAreTheRunsNotThePrior():
    rng = np.random.default_rng(1)
    n = 300
    true1, true2 = rng.normal(1.0, 0.2, n), rng.normal(1.0, 0.2, n)
    prior = priorOf(true1 * (1 + 0.06 * rng.standard_normal(n)),
                    true2 * (1 + 0.06 * rng.standard_normal(n)))
    gains, summary = blindMoveCalib.fitOperatingPoint(opRows(true1, true2, 6), prior=prior)
    err = lambda est, truth: np.sqrt(np.mean((est - truth) ** 2))
    # Six runs of 4% noise: the mean is good to about 0.04 * gain / sqrt(6).
    assert err(gains.landing_gain.to_numpy(), true1) < 0.025
    assert err(gains.target_gain.to_numpy(), true2) < 0.025
    assert summary['prior']['landing']['tau'] == pytest.approx(0.06, abs=0.015)
    assert gains.calibrated.all()


def test_robustMeanOnHandComputedRuns():
    """30 cobras with the same quiet runs, the first with a far last run."""
    quiet = np.array([1.00, 1.02, 0.98, 1.01, 1.00])
    wild = np.array([1.00, 1.02, 0.98, 1.01, 1.40])          # within reject, past huber
    n, m = 30, len(quiet)
    meas = opRows([1.0] * n, [1.0] * n, m, noise=0.0)
    ratios = np.where(meas.cobra_id.to_numpy() == 1, wild[meas.pfs_visit_id - 100],
                      quiet[meas.pfs_visit_id - 100])
    meas['phi_landed'] = meas.phi_start + ratios * (meas.phi_landing_target - meas.phi_start)
    p = blindMoveCalib.OPERATING_POINT
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)

    dof = p['noiseDof']
    fleet = max(np.var(quiet, ddof=1) / blindMoveCalib._medianChi2Ratio(m - 1), p['noiseFloor'] ** 2)
    keep = np.abs(wild - np.median(wild)) <= p['reject'] * np.sqrt(fleet)
    y = wild[keep]
    noise = max((dof * fleet + ((y - y.mean()) ** 2).sum()) / (dof + len(y) - 1), p['noiseFloor'] ** 2)
    mad = (1.4826 * np.median(np.abs(y - np.median(y)))) ** 2
    bound = p['huber'] * np.sqrt((dof * fleet + (len(y) - 1) * mad) / (dof + len(y) - 1))
    expected = np.clip(y, np.median(y) - bound, np.median(y) + bound).mean()
    assert gains.landing_gain[1] == pytest.approx(expected, abs=1e-9)
    assert gains.landing_scatter[1] == pytest.approx(np.sqrt(noise), abs=1e-9)
    assert gains.landing_gain[1] < 1.03                     # the far run did not drag it


def test_twoRunsFarFromThePriorAreTakenAsTheyAre():
    prior = priorOf([1.3] * 20, [1.0] * 20)
    meas = opRows([1.0] * 20, [1.0] * 20, 2, noise=0.02, seed=3)
    gains, summary = blindMoveCalib.fitOperatingPoint(meas, prior=prior, minRuns=2)
    assert summary['landing']['nRejected'] == 0
    assert np.allclose(gains.landing_gain, 1.0, atol=0.05)


def test_withoutPriorTheGainIsTheMeanOfTheRuns():
    meas = opRows([0.8, 1.2], [1.0, 0.9], 4, noise=0.0)
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert np.allclose(gains.landing_gain, [0.8, 1.2]) and np.allclose(gains.target_gain, [1.0, 0.9])


def test_aWildRunIsDropped():
    meas = opRows([1.0] * 20, [1.0] * 20, 6, noise=0.02)
    wild = (meas.pfs_visit_id == 103) & (meas.cobra_id == 5)
    meas.loc[wild, 'phi_landed'] = meas.phi_start[wild] + 2.0 * (meas.phi_landing_target - meas.phi_start)[wild]
    gains, summary = blindMoveCalib.fitOperatingPoint(meas)
    assert gains.landing_gain[5] == pytest.approx(1.0, abs=0.03)
    assert summary['landing']['nRejected'] >= 1


def shifted(meas, cobra, fromVisit, gain):
    late = (meas.pfs_visit_id >= fromVisit) & (meas.cobra_id == cobra)
    x2 = (meas.phi_push_target - meas.phi_landing_target)[late]
    meas.loc[late, 'phi_pushed'] = meas.phi_landed[late] + gain * x2
    return meas


def test_aChangeOfRegimeKeepsTheLaterRuns():
    meas = shifted(opRows([0.5] * 20, [0.5] * 20, 8, noise=0.02), cobra=3, fromVisit=104, gain=0.7)
    gains, summary = blindMoveCalib.fitOperatingPoint(meas)
    assert gains.calibrated[3] and gains.target_gain[3] == pytest.approx(0.7, abs=0.02)
    assert summary['push']['nSplit'] == 1


def test_aChangeOfRegimeTwoRunsBeforeTheEndKeepsThem():
    meas = shifted(opRows([1.0] * 20, [1.0] * 20, 8, noise=0.02), cobra=3, fromVisit=106, gain=1.5)
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert gains.calibrated.all() and gains.target_gain[3] == pytest.approx(1.5, abs=0.05)
    assert gains.n_runs[3] == 2


def drifting(step, n=2000, runs=5, seed=5):
    """Landing runs of white noise 0.03 on a random walk of `step` per run."""
    rng = np.random.default_rng(seed)
    walk = 1.0 + np.cumsum(step * rng.standard_normal((runs, n)), axis=0)
    frames = []
    for run in range(runs):
        r = rows(n, 1.0, 1.0, 0.5, visit=100 + run)
        g = walk[run] + 0.03 * rng.standard_normal(n)
        r['phi_landed'] = r.phi_start + g * (r.phi_landing_target - r.phi_start)
        r['phi_pushed'] = r.phi_landed + (r.phi_push_target - r.phi_landing_target)
        frames.append(r)
    return pd.concat(frames, ignore_index=True)


def test_driftingGainsAreRarelyTakenForAChangeOfRegime():
    """A random walk of 0.022 per run, as measured on 2026-11-04."""
    for runs in (5, 8):
        _, summary = blindMoveCalib.fitOperatingPoint(drifting(0.022, runs=runs))
        assert summary['landing']['nSplit'] / 2000 < 0.005


def test_driftRateIsTheRandomWalkStep():
    _, summary = blindMoveCalib.fitOperatingPoint(drifting(0.022))
    assert summary['landing']['drift'] == pytest.approx(0.022, abs=0.005)
    _, summary = blindMoveCalib.fitOperatingPoint(drifting(0.0))
    assert summary['landing']['drift'] < 0.010


def test_aSteadySlowLandingIsNotAStall():
    meas = opRows([0.4] * 20, [1.0] * 20, 10, noise=0.05, seed=2)
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert (gains.reason != 'stalled landing').all() and gains.calibrated.all()


def test_aStalledLandingIsNotCalibrated():
    meas = opRows([1.0] * 20, [1.0] * 20, 6, noise=0.02)
    stalled = (meas.cobra_id == 7) & meas.pfs_visit_id.isin([101, 104])
    meas.loc[stalled, 'phi_landed'] = meas.phi_start[stalled] + 0.05 * (meas.phi_landing_target - meas.phi_start)[stalled]
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert gains.reason[7] == 'stalled landing'


def test_minRunsCountsEachMove():
    meas = opRows([0.9, 1.1], [1.0, 1.0], 1)
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert not gains.calibrated.any() and (gains.reason == 'fewer than 3 runs').all()
    gains, _ = blindMoveCalib.fitOperatingPoint(meas, minRuns=1)
    assert gains.calibrated.all()


def test_operatingPointIgnoresCobrasNotPushed():
    meas = rows(3, 1.0, 1.0, 0.5)
    meas.loc[1, 'dot_target'] = np.nan
    assert blindMoveCalib.operatingPointOf(meas) == (LANDING, 0.5)
    meas.loc[2, 'dot_target'] = 0.4
    with pytest.raises(ValueError):
        blindMoveCalib.operatingPointOf(meas)


def test_parametersAreStrictJson():
    import json
    text = json.dumps(blindMoveCalib._jsonSafe(dict(a=np.float64(np.nan), b=[np.int64(2), np.inf],
                                                    c=dict(d=np.float32(0.5)))), allow_nan=False)
    assert json.loads(text) == dict(a=None, b=[2, None], c=dict(d=0.5))


def test_unknownParameterIsRefused():
    with pytest.raises(KeyError):
        blindMoveCalib.fitOperatingPoint(opRows([1.0], [1.0], 3), params=dict(huberr=2))


APPROACH = -0.075


def gainsOf(g1, g2, **attrs):
    gains = pd.DataFrame(dict(landing_gain=g1, target_gain=g2, calibrated=True),
                         index=pd.Index(np.arange(1, len(g1) + 1), name='cobra_id'))
    gains.attrs['calibration'] = attrs
    return gains


def aimedRows(aimG1, aimG2, trueG1, trueG2, visit=1, calibId=7):
    """Rows of a turned-dot visit aimed through aimG1, aimG2, of cobras that respond with
    trueG1, trueG2; the push is sized from the landing aimG1 predicts."""
    aimG1, aimG2 = np.asarray(aimG1, float), np.asarray(aimG2, float)
    r = rows(len(aimG1), 1.0, 1.0, 0.5, visit=visit, phiStart=1.2 + (APPROACH - LANDING) * CHORD)
    command = blindMoveCalib.landingCommands(LANDING, APPROACH, aimG1, True)
    r['phi_landing_command'] = r.phi_landing_target + (command - LANDING) * CHORD
    r['phi_landed'] = r.phi_start + np.asarray(trueG1) * (r.phi_landing_command - r.phi_start)
    r['phi_push_start'] = r.phi_start + aimG1 * (r.phi_landing_command - r.phi_start)
    r['phi_push_command'] = r.phi_push_start + (r.phi_push_target - r.phi_push_start) / aimG2
    r['phi_pushed'] = r.phi_landed + np.asarray(trueG2) * (r.phi_push_command - r.phi_push_start)
    r['blind_move_calib_id'] = calibId
    return r


def test_validationOfAnExactCalibration():
    g1, g2 = [0.8, 1.0, 1.3], [0.9, 1.0, 1.2]
    meas = aimedRows(g1, g2, g1, g2)
    passed, stats, failures = blindMoveCalib.validate(meas, gainsOf(g1, g2), 7)
    assert stats['landing']['bias'] == pytest.approx(0.0, abs=1e-9) and stats['landing']['n'] == 3
    assert stats['bias'] == pytest.approx(0.0, abs=1e-9) and passed, failures


def test_aLandingDeeperThanPredictedEndsDeeper():
    meas = aimedRows([1.0] * 3, [1.0] * 3, [1.1] * 3, [1.0] * 3)
    _, stats, _ = blindMoveCalib.validate(meas, gainsOf([1.0] * 3, [1.0] * 3), 7)
    assert stats['bias'] == pytest.approx(0.1 * (LANDING - APPROACH), abs=1e-9)


def test_counterfactualOfTheRunsOwnGainsIsTheRun():
    g1, g2 = [0.7, 1.2, 1.0], [0.9, 1.1, 1.3]
    meas = aimedRows(g1, g2, [0.75, 1.1, 1.0], [0.95, 1.0, 1.3])
    landed, final = blindMoveCalib.counterfactualErrors(meas, gainsOf(g1, g2), aimLanding=True)
    assert np.allclose(landed, blindMoveCalib.landedDepth(meas) - LANDING, atol=1e-12)
    assert np.allclose(final, blindMoveCalib.reachedDepth(meas) - 0.5, atol=1e-12)


def test_counterfactualCapsAPullBackAtTheShallowCap():
    """Predicted past the target from a landing commanded to 0.1, which a cobra of gain 1
    would have stopped at: no pull-back, as `dotMove.blindMoveToDots` would not."""
    meas = rows(1, 3.5, 1.0, 0.5, phiStart=1.2 + (APPROACH - LANDING) * CHORD)
    _, final = blindMoveCalib.counterfactualErrors(meas, gainsOf([3.5], [1.0]), aimLanding=False)
    assert final == pytest.approx([APPROACH + 3.5 * (LANDING - APPROACH) - 0.5], abs=1e-9)


def test_anOperatingPointCalibrationNeedsEveryLandingAimed():
    g = [0.8] * 3
    meas = pd.concat([aimedRows(g, g, g, g, visit=1),
                      aimedRows(g, g, g, g, visit=2).assign(phi_landing_command=lambda r: r.phi_landing_target)])
    _, _, failures = blindMoveCalib.validate(meas, gainsOf(g, g, model=blindMoveCalib.OPERATING_POINT_MODEL), 7)
    assert 'the landing of visits [2] was not aimed' in failures


def test_aCobraNotPushedIsNotTakenForUnaimed():
    g = [0.8, 0.9]
    meas = aimedRows(g, g, g, g)
    meas.loc[1, ['dot_target', 'dot_command', 'phi_push_target', 'phi_push_command']] = np.nan
    assert blindMoveCalib.checkAimedBy(meas, gainsOf(g, g), 7).tolist() == [False, False]


def test_aDegradedCalibrationFailsThePairedCheck():
    """5% per cobra worse than the reference, on its own aimed rows: caught on both moves."""
    rng = np.random.default_rng(4)
    n = 300
    g1, g2 = rng.normal(1.0, 0.15, n), rng.normal(1.0, 0.15, n)
    bad1, bad2 = g1 * (1 + 0.05 * rng.standard_normal(n)), g2 * (1 + 0.05 * rng.standard_normal(n))
    meas = aimedRows(bad1, bad2, g1, g2)
    _, stats, failures = blindMoveCalib.validate(meas, gainsOf(bad1, bad2), 7, reference=gainsOf(g1, g2))
    assert any(f.startswith('counterfactual landing rms') for f in failures)
    assert any(f.startswith('counterfactual final rms') for f in failures)
    meas = aimedRows(g1, g2, g1, g2)
    passed, _, failures = blindMoveCalib.validate(meas, gainsOf(g1, g2), 7, reference=gainsOf(bad1, bad2))
    assert passed, failures


def test_aFewWildCobrasDoNotDecideThePairedCheck():
    assert blindMoveCalib._rms([0.0, 0.0, 1.0], clip=0.1) == pytest.approx(np.sqrt(0.01 / 3))


def test_theCommonModeDoesNotDecideThePairedCheck():
    """2% on every gain is a level, judged by the bias, not a regression."""
    rng = np.random.default_rng(6)
    n = 300
    g1, g2 = rng.normal(1.0, 0.15, n), rng.normal(1.0, 0.15, n)
    meas = aimedRows(g1, g2, g1 * 1.02, g2 * 1.02)
    passed, _, failures = blindMoveCalib.validate(meas, gainsOf(g1, g2), 7,
                                                  reference=gainsOf(g1 * 1.02, g2 * 1.02))
    assert not any(f.startswith('counterfactual') for f in failures), failures


def test_theFleetTrendIsTheSlopeOfTheRunMedians():
    meas = drifting(0.0)
    for run in range(5):
        r = meas.pfs_visit_id == 100 + run
        meas.loc[r, 'phi_landed'] = meas.phi_start[r] + (1 + 0.003 * run) * (meas.phi_landed - meas.phi_start)[r]
    _, summary = blindMoveCalib.fitOperatingPoint(meas)
    assert summary['landing']['trend'] == pytest.approx(0.003, abs=0.001)


def test_counterfactualOfTheRunsOwnGainsIsExact():
    meas = rows(4, 1.2, 0.8, 0.5).assign(phi_landing_command=np.nan, phi_push_start=np.nan,
                                         phi_push_command=np.nan)
    own = pd.DataFrame(dict(landing_gain=1.2, target_gain=0.8, calibrated=True),
                       index=pd.Index(np.arange(1, 5), name='cobra_id'))
    landed, final = blindMoveCalib.counterfactualErrors(meas, own, aimLanding=True,
                                                        approach=-0.075 - 0.1 * 0)
    # Aimed with the cobra's own gains: the push reaches the target exactly.
    assert np.allclose(final, 0.0, atol=1e-9)




def test_aLandingCommandOtherThanTheCalibrationsIsNotAimed():
    gains = pd.DataFrame(dict(landing_gain=[0.8, 1.0], target_gain=[1.0, 1.0], calibrated=[True, True]),
                         index=pd.Index([1, 2], name='cobra_id'))
    meas = rows(2, 1.0, 1.0, 0.5)
    command = blindMoveCalib.landingCommands(LANDING, -0.075, gains.landing_gain.to_numpy(), True)
    meas['phi_landing_command'] = meas.phi_landing_target + (command - LANDING) * CHORD
    meas['phi_push_start'], meas['phi_push_command'] = 1.2, 1.3
    meas['blind_move_calib_id'] = 7
    assert blindMoveCalib.checkAimedBy(meas, gains, 7).tolist() == [False, False]
    meas.loc[0, 'phi_landing_command'] += 0.01
    assert blindMoveCalib.checkAimedBy(meas, gains, 7).tolist() == [True, False]


def test_recordsCarryTheCommands(monkeypatch):
    """Cobra 1 aimed and pushed, cobra 4 not calibrated and not pushed."""
    from ics.cobraCharmer.cobraCoach.engineer import moveDtype
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)
    moves = np.zeros((nCobras, 2), dtype=moveDtype)
    moves['phiAngle'] = [0.1, 0.2]
    moves['detected'] = True
    monkeypatch.setattr(dotMove, 'loadRunInputs', lambda cc, dataDir=None: (moves, np.arange(nCobras), dotIdx))
    monkeypatch.setattr(dotMove, 'rampResidual', lambda db, visit, iteration, n: np.zeros(n))
    monkeypatch.setattr(dotGeometry, 'phiNearTheta', lambda cc, idx, pos, theta: np.full(len(idx), 0.5))
    t = tracker(nCobras, calibrated=[False, True, False, False, False, False])
    t.lastPush = pd.DataFrame(dict(cobraIdx=[1], dotTarget=[0.5], phiPushTarget=[0.45],
                                   phiPushStart=[0.2], phiPushCommand=[0.6], calibrated=[True]))
    rec = dotMove.blindMoveRecords(fakeCc(nCobras), t, dotIdx, 9, None, np.zeros(nCobras, complex), 0.2,
                                   goHome=False, dataDir=None, theta=np.zeros(nCobras),
                                   phiLandingCommand=np.linspace(0, 0.5, nCobras), calibId=7)
    assert rec.phi_landing_command.tolist() == [0.1, 0.4]
    assert rec.phi_push_start.tolist()[0] == 0.2 and rec.phi_push_command.tolist()[0] == 0.6
    assert rec.dot_command.tolist()[0] == 0.5 and np.isnan(rec.phi_push_command.tolist()[1])
    assert rec.blind_move_calib_id[0] == 7 and pd.isna(rec.blind_move_calib_id[1])


def test_leaveOneOutOfAimedVisitsWithExactGainsIsExact():
    g1, g2 = [0.7, 1.0, 1.3, 0.9], [0.8, 1.1, 1.2, 1.0]
    meas = pd.concat([aimedRows(g1, g2, g1, g2, visit=v) for v in range(1, 5)], ignore_index=True)
    errors = blindMoveCalib.leaveOneOut(meas)
    assert len(errors) == 16 and np.allclose(errors, 0.0, atol=1e-9)


def test_theUnaimedComparisonCatchesAWorseLandingGain():
    """Aimed, a wrong landing gain is corrected by the push; unaimed it is what sizes it."""
    rng = np.random.default_rng(7)
    n = 300
    g1, g2 = rng.normal(1.0, 0.15, n), rng.normal(1.0, 0.15, n)
    bad1 = g1 * (1 + 0.08 * rng.standard_normal(n))
    meas = aimedRows(bad1, g2, g1, g2)
    _, stats, failures = blindMoveCalib.validate(meas, gainsOf(bad1, g2), 7, reference=gainsOf(g1, g2),
                                                 alsoUnaimed=True)
    assert 'regressionUnaimed' in stats
    assert any(f.startswith('counterfactual final rms with the landing unaimed') for f in failures)


def test_thePriorIsTheCalibrationTheRowsWereAimedThrough():
    g = [0.8, 0.9]
    assert blindMoveCalib.aimingCalibOf(aimedRows(g, g, g, g, calibId=7)) == 7
    unaimed = aimedRows(g, g, g, g).assign(blind_move_calib_id=pd.array([pd.NA, pd.NA], dtype='Int64'))
    assert blindMoveCalib.aimingCalibOf(unaimed) is None
    with pytest.raises(ValueError):
        blindMoveCalib.aimingCalibOf(pd.concat([aimedRows(g, g, g, g, calibId=7),
                                                aimedRows(g, g, g, g, visit=2, calibId=8)]))


def test_anAimedFastCobraIsKept():
    """Aimed through its gain of 3, a fast cobra commands a short landing and push."""
    g = [1.0] * 29 + [3.0]
    meas = pd.concat([aimedRows(g, g, g, g, visit=v) for v in range(1, 5)], ignore_index=True)
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert gains.calibrated[30] and gains.landing_gain[30] == pytest.approx(3.0)
    assert gains.target_gain[30] == pytest.approx(3.0)


def test_aCobraWithNoUsableRunHasARow():
    meas = opRows([1.0] * 5, [1.0] * 5, 4, noise=0.02)
    meas.loc[meas.cobra_id == 5, 'ramp_residual'] = 0.5
    gains, _ = blindMoveCalib.fitOperatingPoint(meas)
    assert 5 in gains.index and not gains.calibrated[5] and gains.reason[5] == 'fewer than 3 runs'
