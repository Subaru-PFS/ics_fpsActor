"""plateTracker.PlateTracker: the plate applied per frame and its opdb rows; dotGeometry in the plate frame."""
import types

import numpy as np
import pytest

from ics.cobraCharmer import plateFrame
from ics.fpsActor.utils import dotGeometry, plateTracker

N_COBRAS = 60
PLATE = plateFrame.PlateFit(0.040 - 0.010j, (1 + 20e-6) * np.exp(1j * np.deg2rad(10 / 3600)), 0, 0.0)


def reference():
    rng = np.random.default_rng(0)
    ref = np.full(N_COBRAS, np.nan + 0j)
    ref[::2] = 180 * np.sqrt(rng.uniform(0, 1, N_COBRAS // 2)) * np.exp(1j * rng.uniform(0, 2 * np.pi, N_COBRAS // 2))
    return ref


class FakeDb:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def insert_dataframe(self, table, df):
        if self.fail:
            raise RuntimeError('relation does not exist')
        assert table == plateTracker.TABLE
        self.rows.extend(df.to_dict('records'))


def tracker(enabled=True, plateFrames=3, db=None):
    cc = types.SimpleNamespace(calibModel=types.SimpleNamespace(fiberReference=reference()))
    return plateTracker.PlateTracker(cc, enabled, plateFrames=plateFrames,
                                     connectToDB=(lambda: db) if db is not None else None)


def frame(t, visit, i, plate=PLATE, noiseMm=0.0, seed=None):
    rng = np.random.default_rng(seed if seed is not None else i)
    ref = t.cc.calibModel.fiberReference
    measured = np.where(np.isfinite(ref), plate.apply(ref), np.nan)
    measured = measured + noiseMm * (rng.normal(size=N_COBRAS) + 1j * rng.normal(size=N_COBRAS))
    return t(100 * visit + i, measured)


def close(a, b, atol=1e-9):
    return abs(a.a - b.a) < atol and abs(a.b - b.b) < atol


def test_identityUntilEnoughFramesThenThePlate():
    t = tracker()
    assert frame(t, 1000, 0) is plateFrame.IDENTITY
    assert frame(t, 1000, 1) is plateFrame.IDENTITY
    assert close(frame(t, 1000, 2), PLATE)
    assert t.appliedFrames == 3
    frame(t, 1000, 3)
    assert t.appliedFrames == 4                                        # updated every frame


def test_medianOverTheVisitFrames():
    t = tracker()
    for i in range(5):
        frame(t, 1000, i, noiseMm=0.005)
    edge = 200 * np.exp(1j * np.linspace(0, 2 * np.pi, 8))
    assert np.max(np.abs(t.applied.apply(edge) - PLATE.apply(edge))) < 0.004


def test_disabledAppliesTheIdentityAndStillRecords():
    db = FakeDb()
    t = tracker(enabled=False, db=db)
    for i in range(4):
        assert frame(t, 1000, i) is plateFrame.IDENTITY
    assert len(db.rows) == 4
    last = db.rows[-1]
    assert (last['x0'], last['y0'], last['scale'], last['theta'], last['n_frames']) == (0, 0, 1, 0, 0)
    assert last['frame_x0'] == pytest.approx(PLATE.a.real) and last['frame_n_anchors'] == N_COBRAS // 2


def test_newVisitStartsANewStack():
    t = tracker()
    for i in range(3):
        frame(t, 1000, i)
    assert frame(t, 1001, 0) is plateFrame.IDENTITY
    assert t.appliedFrames == 0 and len(t.stack) == 1


def test_unseenFramesAreIgnored():
    t = tracker()
    for i in range(3):
        frame(t, 1000, i)
    assert close(t(100 * 1000 + 3, np.full(N_COBRAS, np.nan + 0j)), PLATE)


def test_failedFitKeepsThePlateOfTheVisit(monkeypatch):
    t = tracker()
    for i in range(3):
        frame(t, 1000, i)
    applied = t.applied
    monkeypatch.setattr(plateTracker.plateFrame, 'fitPlate', lambda *a, **k: None)
    assert frame(t, 1000, 3) is applied
    assert t.appliedFrames == 3


def test_atLeastThreeFrames():
    assert tracker(plateFrames=1).plateFrames == 3
    assert tracker(plateFrames=5).plateFrames == 5


def test_rowsOfAFrameWithoutFit():
    db = FakeDb()
    t = tracker(db=db)
    t(100 * 1000, np.full(N_COBRAS, np.nan + 0j))
    row = db.rows[0]
    assert row['mcs_frame_id'] == 100 * 1000
    assert all(row[f'frame_{k}'] is None for k in ('x0', 'y0', 'scale', 'theta', 'n_anchors', 'rms'))
    assert row['n_frames'] == 0 and row['rms'] is None


def test_databaseFailureStopsRecordingOnly():
    db = FakeDb(fail=True)
    t = tracker(db=db)
    for i in range(plateTracker.MAX_FAILURES - 1):
        plate = frame(t, 1000, i)
    assert t.connectToDB is not None                                   # still trying
    plate = frame(t, 1000, plateTracker.MAX_FAILURES)
    assert close(plate, PLATE)
    assert t.connectToDB is None


def test_oneFailureDoesNotStopRecording():
    db = FakeDb(fail=True)
    t = tracker(db=db)
    frame(t, 1000, 0)
    db.fail = False
    frame(t, 1000, 1)
    assert len(db.rows) == 1 and t.failures == 0


def test_dotAnglesUseTheDotsCarriedByThePlate(monkeypatch):
    """computeDotAngles solves for the dots in the frame of the current centres."""
    n = len(dotGeometry.sgfm)
    xml = dotGeometry.sgfm.x.to_numpy() + 1j * dotGeometry.sgfm.y.to_numpy()
    seen = []

    class Stop(Exception):
        pass

    def positionsToAngles(cobras, positions):
        seen.append(positions.copy())
        raise Stop()

    cc = types.SimpleNamespace(allCobras=np.arange(n), plate=PLATE, xmlCenters=xml,
                               pfi=types.SimpleNamespace(positionsToAngles=positionsToAngles))
    with pytest.raises(Stop):
        dotGeometry.computeDotAngles(cc)
    dots = dotGeometry.sgfm.xDot.to_numpy() + 1j * dotGeometry.sgfm.yDot.to_numpy()
    assert np.allclose(seen[0], PLATE.apply(dots), rtol=0, atol=1e-12)


def test_turnedDotsTurnAboutTheCurrentCentres():
    """Turned dots: the dots carried by the plate, turned about the centres the cobras run."""
    n = len(dotGeometry.sgfm)
    xml = dotGeometry.sgfm.x.to_numpy() + 1j * dotGeometry.sgfm.y.to_numpy()
    current = PLATE.apply(xml) + 0.3                                   # recentred since: not the plate of xml
    seen = []

    class Stop(Exception):
        pass

    def positionsToAngles(cobras, positions):
        seen.append(positions.copy())
        raise Stop()

    cc = types.SimpleNamespace(allCobras=np.arange(n), plate=PLATE, xmlCenters=xml,
                               calibModel=types.SimpleNamespace(centers=current),
                               pfi=types.SimpleNamespace(positionsToAngles=positionsToAngles))
    with pytest.raises(Stop):
        dotGeometry.computeDotAngles(cc, dotThetaOffset=40.0)
    dots = PLATE.apply(dotGeometry.sgfm.xDot.to_numpy() + 1j * dotGeometry.sgfm.yDot.to_numpy())
    assert np.allclose(np.abs(seen[0] - current), np.abs(dots - current), rtol=0, atol=1e-9)
    assert np.allclose(np.angle((seen[0] - current) / (dots - current)), np.deg2rad(40.0), atol=1e-9)
