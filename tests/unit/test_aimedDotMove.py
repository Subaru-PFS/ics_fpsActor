"""The aimed dot moves end to end: landing command, tracker estimate, push steps.

A cobra that moves exactly its landing gain g1 times its landing's map travel and its push
gain g2 times its push's must end at the target depth, whatever happened to the landing:
measured, predicted, or never made because the cobra was frozen before it.  The motor map
is linear (see test_dotState), so the only error is the rounding of steps.
"""
import numpy as np
import pytest

from ics.cobraCharmer.cobraCoach.engineer import moveDtype
from ics.fpsActor.utils import blindMoveCalib, dotGeometry, dotState
from test_dotState import ARM_MM, FakeModel

CENTER, HALF = 1.6, 0.3                  # phi of the dot centre, half its width, radians
APPROACH = dotGeometry.RAMP_APPROACH_FRACTION
LANDING, TARGET = 0.2, 0.5
ROUNDING = 0.01                          # dot fraction a step's rounding can account for


def phiAt(fraction):
    return dotGeometry.computePhiAtFraction(CENTER, HALF, 1, fraction)


def depthOf(phi):
    return 0.5 + (phi - CENTER) / (2 * HALF)


def converge(model, g1, landingCommand, landingSeen, frozen):
    """The approach, measured, then the landing: its moves row and where it really ends."""
    start = phiAt(APPROACH)
    steps = 0 if frozen else dotState.stepsToTarget(model, 0, start, phiAt(landingCommand), 1.0)
    landed = start + g1 * (dotState.mapSpan(model, 0, steps, start) - start) if steps else start
    row = np.zeros((1, 2), dtype=moveDtype)
    row[0]['phiSteps'] = [0, steps]
    row[0]['phiAngle'] = [start, landed]
    row[0]['detected'] = [True, landingSeen and not frozen]
    return row, landed


def push(model, g1, g2, row, landed):
    """Reached phi after a push sized from the tracker of a cobra calibrated with g1, g2."""
    tracker = dotState.DotTracker.fromMoves(model, row, np.array([0]), np.array([g2]),
                                            np.array([ARM_MM]), 1, landingGains=np.array([g1]),
                                            calibrated=np.array([True]))
    steps = dotState.stepsToTarget(model, 0, tracker.phi[0], phiAt(TARGET), tracker.gain[0])
    return landed + g2 * (dotState.mapSpan(model, 0, steps, landed) - landed)


@pytest.mark.parametrize('g1, g2', [(1.0, 1.0), (0.45, 0.6), (1.4, 1.25), (0.8, 1.6)])
def test_aimedLandingLandsAtTheLandingDepth(g1, g2):
    model = FakeModel()
    command = blindMoveCalib.landingCommands(LANDING, APPROACH, g1, True)
    _, landed = converge(model, g1, command, landingSeen=False, frozen=False)
    assert depthOf(landed) == pytest.approx(LANDING, abs=ROUNDING)


@pytest.mark.parametrize('g1, g2', [(1.0, 1.0), (0.45, 0.6), (1.4, 1.25), (0.8, 1.6)])
@pytest.mark.parametrize('aimLanding', [True, False])
@pytest.mark.parametrize('landingSeen, frozen', [(False, False), (True, False), (False, True)])
def test_pushReachesTheTargetWhateverTheLanding(g1, g2, aimLanding, landingSeen, frozen):
    model = FakeModel()
    command = (blindMoveCalib.landingCommands(LANDING, APPROACH, g1, True) if aimLanding
               else LANDING)
    row, landed = converge(model, g1, command, landingSeen, frozen)
    assert depthOf(push(model, g1, g2, row, landed)) == pytest.approx(TARGET, abs=ROUNDING)


def test_uncalibratedCobraIsCommandedTheLandingDepth():
    assert blindMoveCalib.landingCommands(LANDING, APPROACH, 0.5, False) == LANDING


def test_landingCommandStaysInsideTheDot():
    lo, hi = blindMoveCalib.LANDING_COMMAND_RANGE
    assert blindMoveCalib.landingCommands(LANDING, APPROACH, 0.3, True) == pytest.approx(hi)
    assert blindMoveCalib.landingCommands(-0.06, APPROACH, 0.3, True) == pytest.approx(APPROACH + lo)


def test_landingGainOutsideItsBoundsIsNotAimed():
    lo, hi = dotState.GAIN_BOUNDS
    for gain in (lo / 2, hi * 2, np.nan):
        assert blindMoveCalib.landingCommands(LANDING, APPROACH, gain, True) == pytest.approx(LANDING)


def test_frozenCobraIsPushedFromWhereItWasLost():
    """A cobra lost on a ramp row and frozen there: that row is replayed at gain 1, not g2."""
    model = FakeModel()
    g1, g2 = 0.8, 1.5
    start = phiAt(APPROACH - 0.2)
    steps = dotState.stepsToTarget(model, 0, start, phiAt(APPROACH), 1.0)
    lost = start + (dotState.mapSpan(model, 0, steps, start) - start)
    row = np.zeros((1, 3), dtype=moveDtype)
    row[0]['phiSteps'] = [0, steps, 0]
    row[0]['phiAngle'] = [start, lost, lost]
    row[0]['detected'] = [True, False, False]
    assert depthOf(push(model, g1, g2, row, lost)) == pytest.approx(TARGET, abs=ROUNDING)


def test_landingGainOnlyActsOnAnUnseenLanding():
    model = FakeModel()
    row, landed = converge(model, 0.6, LANDING, landingSeen=True, frozen=False)
    seen, _, _ = dotState.estimatePhi(row[0], model, 0, 1.0, ARM_MM, landingGain=0.6)
    assert seen == pytest.approx(landed, abs=1e-4)
    row[0]['detected'][-1] = False
    predicted, _, _ = dotState.estimatePhi(row[0], model, 0, 1.0, ARM_MM, landingGain=0.6)
    assert predicted == pytest.approx(landed, abs=1e-6)
    unaware, _, _ = dotState.estimatePhi(row[0], model, 0, 1.0, ARM_MM)
    assert unaware > predicted + 1e-3


def test_trackerKnowsWhichCobrasAreCalibrated():
    model = FakeModel(nCobras=2)
    rows = np.zeros((2, 2), dtype=moveDtype)
    rows['phiAngle'] = phiAt(APPROACH)
    rows['detected'] = True
    tracker = dotState.DotTracker.fromMoves(model, rows, np.array([0, 1]), np.array([1.2, 1.0]),
                                            np.full(2, ARM_MM), 2, calibrated=np.array([True, False]))
    assert tracker.calibrated.tolist() == [True, False]
    assert tracker.gain.tolist() == [1.2, 1.0]


def fakeCc(model, sent, detected=False):
    import types
    from ics.cobraCharmer import func
    return types.SimpleNamespace(allCobras=np.array([func.Cobra(1, 1)]), calibModel=model,
                                 cobraInfo=dict(detected=np.array([detected])),
                                 pfi=types.SimpleNamespace(moveSteps=lambda c, t, p, phiFast: sent.append(p[0])))


@pytest.fixture
def oneDot(monkeypatch):
    monkeypatch.setattr(dotGeometry, 'computeDotAngles',
                        lambda cc, dotThetaOffset=0.0: (None, np.array([CENTER]), None, None, None,
                                                        np.array([1]), np.array([HALF])))


def calibratedTracker(model, g1, g2, row):
    return dotState.DotTracker.fromMoves(model, row, np.array([0]), np.array([g2]),
                                         np.array([ARM_MM]), 1, landingGains=np.array([g1]),
                                         calibrated=np.array([True]))


def test_staleLowGainsCannotPushPastTheExitCap(oneDot):
    """Calibrated at 0.4 / 0.4, moving the map's travel: the push stops at the deep cap."""
    from ics.fpsActor.utils import dotMove
    model, sent, g = FakeModel(), [], 0.4
    command = blindMoveCalib.landingCommands(LANDING, APPROACH, g, True)
    row, _ = converge(model, 1.0, command, landingSeen=False, frozen=False)   # reverted to 1
    tracker = calibratedTracker(model, g, g, row)
    dotMove.blindMoveToDots(fakeCc(model, sent), tracker, np.array([0]), targetFraction=TARGET)
    end = dotState.mapSpan(model, 0, sent[0], phiAt(command))
    assert depthOf(end) == pytest.approx(dotGeometry.DEPTH_CAP[1], abs=ROUNDING)


def test_frozenCobraOfAnAimedLandingIsPushedInFull(oneDot):
    """Frozen before an aimed landing: the cap counts from where it was lost, not the command."""
    from ics.fpsActor.utils import dotMove
    model, sent, g = FakeModel(), [], 0.7          # capped from its command, not from where it is
    command = blindMoveCalib.landingCommands(LANDING, APPROACH, g, True)
    row, landed = converge(model, g, command, landingSeen=False, frozen=True)
    tracker = calibratedTracker(model, g, g, row)
    dotMove.blindMoveToDots(fakeCc(model, sent), tracker, np.array([0]), targetFraction=TARGET)
    end = landed + g * (dotState.mapSpan(model, 0, sent[0], landed) - landed)
    assert depthOf(end) == pytest.approx(TARGET, abs=ROUNDING)


def test_laterStepsCountTheTravelAlreadySent(oneDot):
    """Flux-scan steps after a capped push cannot carry a reverted cobra past the cap."""
    from ics.fpsActor.utils import dotMove
    model, sent, g = FakeModel(), [], 0.4
    command = blindMoveCalib.landingCommands(LANDING, APPROACH, g, True)
    row, _ = converge(model, 1.0, command, landingSeen=False, frozen=False)
    tracker = calibratedTracker(model, g, g, row)
    cc = fakeCc(model, sent)
    dotMove.blindMoveToDots(cc, tracker, np.array([0]), targetFraction=TARGET)
    for _ in range(6):
        dotMove.blindMoveToDots(cc, tracker, np.array([0]), deltaFraction=0.05)
    end = phiAt(command)
    for steps in sent:
        end = dotState.mapSpan(model, 0, steps, end)
    assert depthOf(end) <= dotGeometry.DEPTH_CAP[1] + ROUNDING


def seenAt(fraction):
    row = np.zeros((1, 2), dtype=moveDtype)
    row[0]['phiAngle'] = phiAt(fraction)
    row[0]['detected'] = True
    return row


def test_aCobraPastTheCapIsPulledBack(oneDot):
    from ics.fpsActor.utils import dotMove
    model, sent, g = FakeModel(), [], 0.8
    tracker = calibratedTracker(model, g, g, seenAt(0.95))
    dotMove.blindMoveToDots(fakeCc(model, sent), tracker, np.array([0]), targetFraction=TARGET,
                            requireHidden=False)
    start = phiAt(0.95)
    assert sent[0] * np.sign(phiAt(1) - phiAt(0)) < 0
    assert depthOf(start + g * (dotState.mapSpan(model, 0, sent[0], start) - start)) == pytest.approx(TARGET, abs=ROUNDING)


def test_fluxStepsAreSizedAndFollowedWithThePushGain(oneDot):
    from ics.fpsActor.utils import dotMove
    model, sent, g = FakeModel(), [], 0.5
    tracker = calibratedTracker(model, 1.0, g, seenAt(0.3))
    dotMove.blindMoveToDots(fakeCc(model, sent), tracker, np.array([0]), deltaFraction=0.05,
                            requireHidden=False)
    assert depthOf(dotState.mapSpan(model, 0, sent[0], phiAt(0.3))) == pytest.approx(0.4, abs=ROUNDING)
    assert depthOf(tracker.phi[0]) == pytest.approx(0.35, abs=ROUNDING)


def test_aPullBackStopsAtTheShallowCap(oneDot):
    """Seen at 0.9 with a stale push gain of 0.4: moving the map's travel, it stops at 0.1."""
    from ics.fpsActor.utils import dotMove
    model, sent = FakeModel(), []
    tracker = calibratedTracker(model, 1.0, 0.4, seenAt(0.9))
    dotMove.blindMoveToDots(fakeCc(model, sent), tracker, np.array([0]), targetFraction=TARGET,
                            requireHidden=False)
    assert depthOf(dotState.mapSpan(model, 0, sent[0], phiAt(0.9))) == pytest.approx(
        dotGeometry.DEPTH_CAP[0], abs=ROUNDING)
