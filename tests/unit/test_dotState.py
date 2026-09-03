"""Tests for dotState, aimed at the errors this module exists to avoid.

The gain divides rather than multiplies; a cobra that is not seen contributes nothing
even though its recorded angle looks plausible; the slow map is used throughout.  Each
of those is a wrong answer that looks entirely reasonable in the output, so each gets a
test that fails loudly rather than a comment.

The fake motor map is linear, so every expectation below is exact arithmetic rather than
a tolerance chosen to make the test pass.
"""
import os
import time

import numpy as np
import pytest

from ics.cobraCharmer.cobraCoach.engineer import moveDtype
from ics.fpsActor.utils import dotState

N_BINS = 50
BIN_RAD = np.deg2rad(3.6)
STEPS_PER_BIN = 50.0
RAD_PER_STEP = BIN_RAD / STEPS_PER_BIN

ANGLES = np.arange(N_BINS + 1) * BIN_RAD
SLOW = np.arange(N_BINS + 1) * STEPS_PER_BIN

ARM_MM = 2.3


class FakeModel:
    """Only the arrays dotState is allowed to read."""

    def __init__(self, pos=SLOW, neg=SLOW, angles=ANGLES, nCobras=1):
        self.phiOffsets = np.tile(angles, (nCobras, 1))
        self.posPhiSlowSteps = np.tile(pos, (nCobras, 1))
        self.negPhiSlowSteps = np.tile(neg, (nCobras, 1))


class FastMapTrap(FakeModel):
    """Reading a fast map is a test failure, not a wrong number."""

    @property
    def posPhiSteps(self):
        raise AssertionError('dotState read the fast motor map')

    @property
    def negPhiSteps(self):
        raise AssertionError('dotState read the fast motor map')


def makeRow(phi, steps, detected):
    """One cobra's row of a moves array."""
    row = np.zeros(len(steps), dtype=moveDtype)
    row['phiAngle'] = phi
    row['phiSteps'] = steps
    row['detected'] = detected
    return row


@pytest.fixture
def model():
    return FakeModel()


# ── mapSpan ──────────────────────────────────────────────────────────────────

def test_zeroStepsIsIdentity(model):
    for angle in (0.0, 0.5, 1.2, ANGLES[-1]):
        assert dotState.mapSpan(model, 0, 0, angle) == pytest.approx(angle, abs=1e-12)


def test_stepsAreAdditive(model):
    once = dotState.mapSpan(model, 0, 300, 0.5)
    twice = dotState.mapSpan(model, 0, 200, once)
    assert twice == pytest.approx(dotState.mapSpan(model, 0, 500, 0.5), abs=1e-12)


def test_clipsRatherThanExtrapolating(model):
    assert dotState.mapSpan(model, 0, 10 ** 9, 1.0) == pytest.approx(ANGLES[-1])
    assert dotState.mapSpan(model, 0, -10 ** 9, 1.0) == pytest.approx(ANGLES[0])


def test_startOutsideTheMapIsNaN(model):
    assert np.isnan(dotState.mapSpan(model, 0, 100, ANGLES[-1] + 1.0))
    assert np.isnan(dotState.mapSpan(model, 0, 100, np.nan))


def test_directionSelectsTheTable():
    """A cobra that costs twice the steps closing must move half as far."""
    model = FakeModel(neg=2 * SLOW)
    opening = dotState.mapSpan(model, 0, +400, 0.5) - 0.5
    closing = 0.5 - dotState.mapSpan(model, 0, -400, 0.5)
    assert abs(opening) == pytest.approx(2 * abs(closing), rel=1e-9)


def test_neverReadsTheFastMap():
    dotState.mapSpan(FastMapTrap(), 0, 300, 0.5)


def test_flatMapIsRejected():
    """A plateau makes the table non-invertible; np.interp would silently return its edge."""
    flat = SLOW.copy()
    flat[20:25] = flat[20]
    assert np.isnan(dotState.mapSpan(FakeModel(pos=flat), 0, 400, 0.5))


# ── gainFromMoves ────────────────────────────────────────────────────────────

def test_gainRecoversAKnownValue(model):
    steps = np.array([0, 400, 400, 400, 400, 400])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP * 1.30
    row = makeRow(phi, steps, np.ones(len(steps), bool))
    assert dotState.gainFromMoves(row, model, 0) == pytest.approx(1.30, rel=1e-3)


def test_undetectedAngleIsIgnored(model):
    """The recorded angle of a hidden cobra is the dot-centre artefact, not a measurement.

    Using it here would drag the gain toward the artefact and look entirely plausible.
    """
    steps = np.array([0, 400, 400, 400, 400, 400, 400])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP * 1.30
    phi[-1] = phi[-2] + 0.5 * 400 * RAD_PER_STEP        # would read as gain 0.5
    row = makeRow(phi, steps, [True] * 6 + [False])
    assert dotState.gainFromMoves(row, model, 0) == pytest.approx(1.30, rel=0.05)


def test_backwardsCobraGivesNegativeGain(model):
    """Signed, not absolute: a cobra moving against its command must be visible as such."""
    steps = np.array([0, -400, -400, -400, -400])
    phi = 1.20 + np.cumsum(steps) * RAD_PER_STEP * -1.0   # moving the wrong way
    row = makeRow(phi, steps, np.ones(len(steps), bool))
    assert dotState.gainFromMoves(row, model, 0) < 0


def test_closingCobraGivesPositiveGain(model):
    steps = np.array([0, -400, -400, -400, -400])
    phi = 1.20 + np.cumsum(steps) * RAD_PER_STEP
    row = makeRow(phi, steps, np.ones(len(steps), bool))
    assert dotState.gainFromMoves(row, model, 0) == pytest.approx(1.0, rel=1e-3)


def test_lastMoveBeforeGoingDarkIsDropped(model):
    """Surviving that move is anti-correlated with gain, so it biases the estimate low."""
    steps = np.array([0] + [400] * 8)
    phi = np.empty(len(steps))
    phi[0] = 0.60
    for i in range(1, len(steps) - 1):
        phi[i] = phi[i - 1] + 1.20 * 400 * RAD_PER_STEP
    phi[-1] = phi[-2] + 0.40 * 400 * RAD_PER_STEP        # the slow, surviving move
    row = makeRow(phi, steps, [True] * 8 + [False])
    assert dotState.gainFromMoves(row, model, 0) == pytest.approx(1.20, rel=0.05)


def test_tinyMovesAreExcluded(model):
    """Below ~10 steps quantisation and backlash dominate the ratio."""
    steps = np.array([0, 2, 3, 2, 3, 2])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP
    row = makeRow(phi, steps, np.ones(len(steps), bool))
    assert np.isnan(dotState.gainFromMoves(row, model, 0))


def test_tooFewPairsGivesNaN(model):
    steps = np.array([0, 400, 400])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP
    row = makeRow(phi, steps, [True, True, False])
    assert np.isnan(dotState.gainFromMoves(row, model, 0))


@pytest.fixture
def applyGain(monkeypatch):
    """Pin the measured gain on, so these tests do not depend on the module default."""
    monkeypatch.setattr(dotState, 'APPLY_MEASURED_GAIN', True)


def test_measuredGainIsNotAppliedWhileTheOnTimeLoopOwnsIt(monkeypatch):
    """Both corrections take out the same discrepancy, so only one may be applied."""
    monkeypatch.setattr(dotState, 'APPLY_MEASURED_GAIN', False)
    assert dotState.usableGain(1.078) == dotState.DEFAULT_GAIN


def test_unusableGainFallsBackToOne(monkeypatch):
    monkeypatch.setattr(dotState, 'APPLY_MEASURED_GAIN', True)
    for gain in (np.nan, 0.0, -1.2, 0.1, 5.0):
        assert dotState.usableGain(gain) == dotState.DEFAULT_GAIN
    assert dotState.usableGain(1.078) == pytest.approx(1.078)


# ── estimatePhi ──────────────────────────────────────────────────────────────

def test_estimateFollowsTheMeasurementWhileVisible(model):
    steps = np.array([0, 400, 400, 400])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP
    row = makeRow(phi, steps, np.ones(len(steps), bool))
    estimate, _, blind = dotState.estimatePhi(row, model, 0, 1.0, ARM_MM)
    assert estimate == pytest.approx(phi[-1], abs=2e-3)
    assert blind == 0


def test_estimateCarriesOnWhenTheCobraVanishes(model):
    steps = np.array([0, 400, 400, 400])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP
    phi[-1] = 0.60                                        # the artefact
    row = makeRow(phi, steps, [True, True, True, False])
    estimate, variance, blind = dotState.estimatePhi(row, model, 0, 1.0, ARM_MM)
    # The update does not snap exactly onto the measurement -- K is near but not one --
    # so the propagated estimate sits close to, not on, the artefact-free prediction.
    expected = phi[2] + 400 * RAD_PER_STEP
    assert estimate == pytest.approx(expected, abs=0.05 * 400 * RAD_PER_STEP)
    assert blind == 1
    assert variance > 0


def test_uncertaintyGrowsWithEachBlindMove(model):
    steps = np.array([0, 400, 400, 400, 400])
    phi = np.full(len(steps), 0.60)
    short = makeRow(phi, steps, [True, True, True, False, False])
    long_ = makeRow(phi, steps, [True, True, False, False, False])
    _, vShort, nShort = dotState.estimatePhi(short, model, 0, 1.0, ARM_MM)
    _, vLong, nLong = dotState.estimatePhi(long_, model, 0, 1.0, ARM_MM)
    assert nLong > nShort
    assert vLong > vShort


def test_iterationsThatCommandNothingAddNoUncertainty(model):
    steps = np.array([0, 400, 0, 0])
    phi = np.array([0.60, 0.60 + 400 * RAD_PER_STEP, 0.0, 0.0])
    row = makeRow(phi, steps, [True, True, False, False])
    _, _, blind = dotState.estimatePhi(row, model, 0, 1.0, ARM_MM)
    assert blind == 0


def test_neverSeenGivesNaN(model):
    steps = np.array([0, 400, 400])
    row = makeRow(np.zeros(3), steps, [False, False, False])
    estimate, _, _ = dotState.estimatePhi(row, model, 0, 1.0, ARM_MM)
    assert np.isnan(estimate)


# ── stepsToTarget ────────────────────────────────────────────────────────────

def test_unbelievableGainLeavesTheMapAlone(model):
    """A missing gain, or one outside GAIN_BOUNDS, sizes the move from the map alone."""
    target = 0.60 + 500 * RAD_PER_STEP
    lo, hi = dotState.GAIN_BOUNDS
    for gain in (np.nan, lo / 2, hi * 2):
        assert dotState.stepsToTarget(model, 0, 0.60, target, gain) == 500


def test_dividesByTheGain(model, applyGain):
    """A cobra that overshoots the map needs FEWER steps, not more."""
    target = 0.60 + 500 * RAD_PER_STEP
    assert dotState.stepsToTarget(model, 0, 0.60, target, 1.25) == 400
    assert dotState.stepsToTarget(model, 0, 0.60, target, 1.25) != 625


def test_undershootingCobraNeedsMoreSteps(model, applyGain):
    target = 0.60 + 500 * RAD_PER_STEP
    assert dotState.stepsToTarget(model, 0, 0.60, target, 0.8) == 625


def test_closingMoveIsNegative(model):
    start = 1.20
    target = start - 500 * RAD_PER_STEP
    assert dotState.stepsToTarget(model, 0, start, target, 1.0) == -500


def test_roundTripAgainstTheProcessModel(model, applyGain):
    """stepsToTarget and estimatePhi must agree about what the gain does."""
    start, target = 0.60, 0.60 + 500 * RAD_PER_STEP
    for gain in (0.6, 0.8, 1.0, 1.078, 1.25, 1.6):
        steps = dotState.stepsToTarget(model, 0, start, target, gain)
        reached = start + gain * (dotState.mapSpan(model, 0, steps, start) - start)
        assert reached == pytest.approx(target, abs=2 * RAD_PER_STEP)


def test_collapsedGainCannotSizeAHugeMove(model):
    """The 2026-07-24 failure: a collapsed speed asked for 73919 steps."""
    target = 0.60 + 300 * RAD_PER_STEP
    steps = dotState.stepsToTarget(model, 0, 0.60, target, 1e-6)
    assert abs(steps) <= dotState.STEP_CEILING * 300 + 1


def test_alreadyThereIsZero(model):
    assert dotState.stepsToTarget(model, 0, 0.60, 0.60, 1.0) == 0


def test_unusableInputGivesZero(model):
    assert dotState.stepsToTarget(model, 0, np.nan, 0.60, 1.0) == 0
    assert dotState.stepsToTarget(model, 0, 0.60, np.nan, 1.0) == 0
    assert dotState.stepsToTarget(model, 0, 0.60, ANGLES[-1] + 1.0, 1.0) == 0


# ── loadRecentGains ──────────────────────────────────────────────────────────

def makeRun(root, name, cobraIds, gain, ageDays=0.0, withMapping=True, nIter=6):
    """A run directory carrying a move history with a known gain."""
    data = root / name / 'data'
    data.mkdir(parents=True)

    steps = np.array([0] + [400] * (nIter - 1))
    rows = np.zeros((len(cobraIds), nIter), dtype=moveDtype)
    for row in range(len(cobraIds)):
        rows[row]['phiSteps'] = steps
        rows[row]['phiAngle'] = 0.60 + np.cumsum(steps) * RAD_PER_STEP * gain
        rows[row]['detected'] = True
    np.save(data / 'moves.npy', rows)
    if withMapping:
        np.savez(data / 'cobra_filtering.npz', final_moving_cobras=np.asarray(cobraIds))

    when = time.time() - ageDays * 24 * 3600
    for path in (data / 'moves.npy', data / 'cobra_filtering.npz'):
        if path.exists():
            os.utime(path, (when, when))
    return data


def test_recentGainsAverageOverRuns(tmp_path, model):
    for i, gain in enumerate((1.10, 1.20, 1.30)):
        makeRun(tmp_path, f'20260808_00{i}', [0], gain, ageDays=0.01 * i)
    gains = dotState.loadRecentGains(str(tmp_path), 1, model)
    assert gains[0] == pytest.approx(1.20, rel=0.02)


def test_staleRunsAreExcluded(tmp_path, model):
    """A run from a previous night measured a machine that no longer exists."""
    makeRun(tmp_path, '20260808_000', [0], 1.10, ageDays=0.01)
    makeRun(tmp_path, '20260805_000', [0], 1.90, ageDays=3.0)
    gains = dotState.loadRecentGains(str(tmp_path), 1, model)
    assert gains[0] == pytest.approx(1.10, rel=0.02)


def test_runWithoutTheCobraMappingIsSkipped(tmp_path, model):
    """moves.npy records rows in commanded order and nothing else.

    Without the ids that go with them the rows cannot be attributed, and guessing would
    apply one cobra's gain to another.
    """
    makeRun(tmp_path, '20260808_000', [0], 1.90, ageDays=0.01, withMapping=False)
    makeRun(tmp_path, '20260808_001', [0], 1.10, ageDays=0.02)
    gains = dotState.loadRecentGains(str(tmp_path), 1, model)
    assert gains[0] == pytest.approx(1.10, rel=0.02)


def test_rowsAreAttributedToTheirOwnCobra(tmp_path, model):
    """Rows are positions within the commanded set, not cobra ids."""
    data = tmp_path / '20260808_000' / 'data'
    data.mkdir(parents=True)
    steps = np.array([0] + [400] * 5)
    rows = np.zeros((2, 6), dtype=moveDtype)
    for row, gain in enumerate((1.10, 1.60)):
        rows[row]['phiSteps'] = steps
        rows[row]['phiAngle'] = 0.60 + np.cumsum(steps) * RAD_PER_STEP * gain
        rows[row]['detected'] = True
    np.save(data / 'moves.npy', rows)
    np.savez(data / 'cobra_filtering.npz', final_moving_cobras=np.array([3, 1]))

    gains = dotState.loadRecentGains(str(tmp_path), 5, FakeModel(nCobras=5))
    assert gains[3] == pytest.approx(1.10, rel=0.02)
    assert gains[1] == pytest.approx(1.60, rel=0.02)
    assert np.isnan(gains[0]) and np.isnan(gains[2]) and np.isnan(gains[4])


def test_mismatchedRowCountIsSkipped(tmp_path, model):
    """A motor-map or convergence-test run has a moves.npy of another shape entirely."""
    data = tmp_path / '20260808_000' / 'data'
    data.mkdir(parents=True)
    rows = np.zeros((4, 6), dtype=moveDtype)
    rows['detected'] = True
    np.save(data / 'moves.npy', rows)
    np.savez(data / 'cobra_filtering.npz', final_moving_cobras=np.array([0, 1]))
    assert np.isnan(dotState.loadRecentGains(str(tmp_path), 3, FakeModel(nCobras=3))).all()


def test_missingRootIsNotFatal(model):
    gains = dotState.loadRecentGains('/no/such/root', 4, FakeModel(nCobras=4))
    assert gains.shape == (4,) and np.isnan(gains).all()


def test_reportsWhichRunsItUsed(tmp_path, model):
    class Cmd:
        def __init__(self):
            self.informs, self.warns = [], []

        def inform(self, text):
            self.informs.append(text)

        def warn(self, text):
            self.warns.append(text)

    makeRun(tmp_path, '20260808_000', [0], 1.10, ageDays=0.01)
    cmd = Cmd()
    dotState.loadRecentGains(str(tmp_path), 1, model, cmd=cmd)
    assert any('20260808_000' in m for m in cmd.informs)


# ── DotTracker ───────────────────────────────────────────────────────────────

def makeTracker(model, phi=0.60, gain=1.0, nCobras=1):
    return dotState.DotTracker(model,
                               phi=np.full(nCobras, phi),
                               variance=np.full(nCobras, 1e-8),
                               gain=np.full(nCobras, gain),
                               armLength=np.full(nCobras, ARM_MM))


def test_appliedAdvancesTheEstimate(model):
    tracker = makeTracker(model)
    tracker.applied([0], [400])
    assert tracker.phi[0] == pytest.approx(0.60 + 400 * RAD_PER_STEP, abs=1e-9)


def test_appliedScalesByTheGain(model):
    tracker = makeTracker(model, gain=1.30)
    tracker.applied([0], [400])
    assert tracker.phi[0] == pytest.approx(0.60 + 1.30 * 400 * RAD_PER_STEP, abs=1e-9)


def test_appliedIsANoOpForZeroSteps(model):
    tracker = makeTracker(model)
    before = tracker.phi[0], tracker.variance[0]
    tracker.applied([0], [0])
    assert (tracker.phi[0], tracker.variance[0]) == before


def test_uncertaintyGrowsWithEachCommandedMove(model):
    tracker = makeTracker(model)
    first = tracker.variance[0]
    tracker.applied([0], [400])
    second = tracker.variance[0]
    tracker.applied([0], [400])
    assert second > first and tracker.variance[0] > second


def test_successiveMovesAreIncremental(model):
    """The scan steps on from where it left the cobra, not from where it started.

    Sizing each step from the original position would re-command the whole distance
    every flat and drive the cobra out the far side of its dot.
    """
    tracker = makeTracker(model)
    target = np.array([0.60])
    steps = []
    for _ in range(4):
        target = target + 100 * RAD_PER_STEP
        step = tracker.stepsTo([0], target)
        tracker.applied([0], step)
        steps.append(int(step[0]))
    assert steps == [100, 100, 100, 100]
    assert tracker.phi[0] == pytest.approx(0.60 + 400 * RAD_PER_STEP, abs=1e-6)


def test_observeIgnoresUndetectedCobras(model):
    """The recorded angle of a hidden cobra is the dot-centre artefact."""
    tracker = makeTracker(model)
    tracker.observe(np.array([0.10]), np.array([False]))
    assert tracker.phi[0] == pytest.approx(0.60)


def test_observePullsTowardTheMeasurement(model):
    tracker = makeTracker(model)
    tracker.variance[0] = 1e-4               # a stale estimate
    tracker.observe(np.array([0.70]), np.array([True]))
    assert 0.60 < tracker.phi[0] <= 0.70
    assert tracker.variance[0] < 1e-4


def test_observeAdoptsTheMeasurementWhenNothingIsKnown(model):
    tracker = makeTracker(model)
    tracker.phi[0] = np.nan
    tracker.observe(np.array([0.70]), np.array([True]))
    assert tracker.phi[0] == pytest.approx(0.70)


def test_unknownCobrasCannotBeStepped(model):
    tracker = makeTracker(model)
    tracker.phi[0] = np.nan
    assert tracker.stepsTo([0], np.array([0.90]))[0] == 0
    tracker.applied([0], [400])
    assert np.isnan(tracker.phi[0])


def test_fromMovesSeedsFromTheConvergence(model):
    steps = np.array([0, 400, 400, 400])
    phi = 0.60 + np.cumsum(steps) * RAD_PER_STEP
    rows = np.zeros((1, len(steps)), dtype=moveDtype)
    rows[0]['phiSteps'] = steps
    rows[0]['phiAngle'] = phi
    rows[0]['detected'] = True

    tracker = dotState.DotTracker.fromMoves(model, rows, np.array([0]), np.array([1.0]),
                                            np.array([ARM_MM]), 1)
    assert tracker.phi[0] == pytest.approx(phi[-1], abs=2e-3)
