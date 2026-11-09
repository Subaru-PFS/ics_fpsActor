"""The rotation a boresight needs, and how a measurement is written."""
from datetime import datetime, timezone

import pandas as pd
import pytest

import ics.fpsActor.boresightMeasurements as boresightMeasure


@pytest.fixture
def insRot(monkeypatch):
    """Give each frame the insrot of a dict, in place of mcs_exposure."""
    angles = {}
    monkeypatch.setattr(boresightMeasure, 'loadTelescopeParametersFromDB',
                        lambda db, frameId: (0., angles[frameId]))
    return angles


@pytest.mark.parametrize('angles', [(-15., 15.), (0., 30.), (0., 0.1, 30.), (-177., 177.)])
def test_twoAnglesAccepted(insRot, angles):
    insRot.update(enumerate(angles))
    boresightMeasure.checkRotation(None, list(insRot))


@pytest.mark.parametrize('angles', [(15., 15.), (15., 15.2), (-179.5, 179.5), (30., 31., 32.),
                                    (0., float('nan'))])
def test_oneAngleRefused(insRot, angles):
    """Frames within MIN_ROTATION of each other, across the wrap included, give no centre; nor
    does an unknown angle."""
    insRot.update(enumerate(angles))
    with pytest.raises(ValueError, match='two angles'):
        boresightMeasure.checkRotation(None, list(insRot))


def test_refusedBeforeMeasuring(insRot, monkeypatch):
    """calcBoresight refuses one angle before reading a centroid or writing anything."""
    insRot.update({15255000: 15., 15255001: 15.})

    def neverCalled(*args, **kwargs):
        raise AssertionError('reached past the rotation check')

    monkeypatch.setattr(boresightMeasure, 'loadCentroidsFromDB', neverCalled)
    monkeypatch.setattr(boresightMeasure, 'writeBoresightToDB', neverCalled)
    with pytest.raises(ValueError, match='two angles'):
        boresightMeasure.calcBoresight(None, None, [15255000, 15255001], 152550, writeToDB=True)


class FakeDB:
    """Hands out sequence values and keeps what is inserted, per table."""

    def __init__(self):
        self.inserted = {}

    def query_scalar(self, sql):
        assert 'mcs_boresight_frame_set_id_seq' in sql
        return 7

    def insert(self, table, df):
        self.inserted[table] = df


def test_measurementWrittenWithItsFrames():
    db = FakeDB()
    before = datetime.now(timezone.utc)
    boresightMeasure.writeBoresightToDB(db, 152550, (4441.2, 2839.2), [15255000, 15255001])

    frameSet = db.inserted['mcs_boresight_frame_set']
    assert frameSet.mcs_boresight_frame_set_id.tolist() == [7, 7]
    assert frameSet.mcs_frame_id.tolist() == [15255000, 15255001]

    [meas] = db.inserted['mcs_boresight_meas'].to_dict('records')
    assert meas['pfs_visit_id'] == 152550
    assert (meas['mcs_boresight_x_pix'], meas['mcs_boresight_y_pix']) == (4441.2, 2839.2)
    assert meas['mcs_boresight_frame_set_id'] == 7
    calculatedAt = pd.Timestamp(meas['calculated_at'])
    assert calculatedAt.tzinfo is not None
    assert abs((calculatedAt - pd.Timestamp(before)).total_seconds()) < 60
