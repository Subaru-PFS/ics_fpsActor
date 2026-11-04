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


def tracker(nCobras):
    return types.SimpleNamespace(phi=np.zeros(nCobras), gain=np.ones(nCobras),
                                 applied=lambda m, s: None, lastPush=None)


def test_pushUsesTheAimedCommand(monkeypatch):
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)
    monkeypatch.setattr(dotMove, 'aimThroughCalibration',
                        lambda cc, wanted, *a: (wanted - 0.03, np.ones(len(wanted), bool)))
    t = tracker(nCobras)
    dotMove.blindMoveToDots(fakeCc(nCobras), t, dotIdx, targetFraction=0.5, requireHidden=False,
                            gains='calib', landingFraction=0.12)
    assert np.allclose(t.lastPush.dotTarget, 0.5) and np.allclose(t.lastPush.dotCommand, 0.47)
    assert np.allclose(t.lastPush.phiPushTarget, 2 * (0.47 - 0.5))


def test_aimingFaultLeavesThePushUnaimed(monkeypatch):
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)

    def broken(*a):
        raise ValueError('cannot reindex on an axis with duplicate labels')

    monkeypatch.setattr(dotMove, 'aimThroughCalibration', broken)
    warnings = []
    cmd = types.SimpleNamespace(inform=lambda text: None, warn=warnings.append)
    t = tracker(nCobras)
    assert dotMove.blindMoveToDots(fakeCc(nCobras), t, dotIdx, cmd=cmd, targetFraction=0.5,
                                   requireHidden=False, gains='calib') == len(dotIdx)
    assert np.allclose(t.lastPush.dotCommand, 0.5) and 'unaimed' in warnings[0]


def test_fluxStepIsNeverAimed(monkeypatch):
    nCobras, dotIdx = 6, np.array([1, 4])
    patchGeometry(monkeypatch, nCobras)
    called = []
    monkeypatch.setattr(dotMove, 'aimThroughCalibration', lambda *a: called.append(1))
    dotMove.blindMoveToDots(fakeCc(nCobras), tracker(nCobras), dotIdx, deltaFraction=0.05,
                            requireHidden=False, gains='calib')
    assert called == []
