"""Loading a boresight's frames from opdb, reporting the measurement and writing it."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import ics.fpsActor.boresightMeasurements as boresightMeasure

CENTRE = (4438.49, 2839.29)


def loop(insRots, firstFrameId=15395100, nPoints=300, seed=1):
    """mcs_exposure rows and identified spots of fixed points turned about CENTRE by -insrot."""
    rng = np.random.default_rng(seed)
    r = 2700 * np.sqrt(rng.uniform(0, 1, nPoints))
    phi = rng.uniform(0, 2 * np.pi, nPoints)
    frames, points = [], []
    for k, insRot in enumerate(insRots):
        frameId = firstFrameId + k
        angle = phi - np.deg2rad(insRot)
        x = CENTRE[0] + r * np.cos(angle) + rng.normal(0, 0.1, nPoints)
        y = CENTRE[1] + r * np.sin(angle) + rng.normal(0, 0.1, nPoints)
        frames.append(dict(mcs_frame_id=frameId, insrot=insRot, altitude=60.0))
        points.append(
            pd.DataFrame(dict(mcs_frame_id=frameId, kind='cobra', id=np.arange(nPoints) + 1, x=x, y=y))
        )
    return pd.DataFrame(frames), pd.concat(points, ignore_index=True)


class FakeDB:
    """Serves mcs_exposure and the spots of some frames; keeps what is inserted, per table."""

    def __init__(self, frames, points):
        self.frames, self.points = frames, points
        self.inserted = {}
        self.queries = []

    def query_dataframe(self, sql, params=None):
        # pfs_utils DB.query_dataframe binds :name only; a %(name)s reaches the database as is.
        assert '%(' not in sql and ':frameIds' in sql
        self.queries.append(sql)
        wanted = params['frameIds']
        if 'FROM mcs_exposure' in sql:
            return self.frames[self.frames.mcs_frame_id.isin(wanted)].iloc[::-1]
        return self.points[self.points.mcs_frame_id.isin(wanted)]

    def query_scalar(self, sql):
        assert 'mcs_boresight_frame_set_id_seq' in sql
        return 7

    def insert(self, table, df):
        self.inserted[table] = df


class Cmd:
    def __init__(self):
        self.informs, self.warnings = [], []

    def inform(self, text):
        self.informs.append(text)

    def warn(self, text):
        self.warnings.append(text)


def test_loadFramesInOrderGiven():
    """Frames come back in the order given, with their spots, NaN centroids dropped."""
    frames, points = loop([-15, 15, 45])
    points.loc[0, 'x'] = np.nan
    db = FakeDB(frames, points)
    loaded, spots = boresightMeasure.loadFrames(db, [15395102, 15395100, 15395101])
    assert loaded.mcs_frame_id.tolist() == [15395102, 15395100, 15395101]
    assert len(spots) == len(points) - 1
    assert len(db.queries) == 2


@pytest.mark.parametrize(
    'frameIds,match', [([15395100, 15395100], 'more than once'), ([15395100, 15395199], 'no mcs_exposure')]
)
def test_loadFramesRefused(frameIds, match):
    with pytest.raises(ValueError, match=match):
        boresightMeasure.loadFrames(FakeDB(*loop([-15, 15])), frameIds)


def test_loopMeasuredAndWritten():
    """A loop is measured, reported frame by frame and written with its frames and uncertainty."""
    frames, points = loop(np.arange(-165, 166, 30))
    db, cmd = FakeDB(frames, points), Cmd()
    before = datetime.now(timezone.utc)
    result = boresightMeasure.calcBoresight(cmd, db, frames.mcs_frame_id.tolist(), 153951, writeToDB=True)

    assert abs(result.x - CENTRE[0]) < 0.05 and abs(result.y - CENTRE[1]) < 0.05
    assert sum('frame 1539' in text for text in cmd.informs) == 12
    assert not cmd.warnings

    frameSet = db.inserted['mcs_boresight_frame_set']
    assert frameSet.mcs_boresight_frame_set_id.tolist() == [7] * 12
    assert frameSet.mcs_frame_id.tolist() == frames.mcs_frame_id.tolist()

    [meas] = db.inserted['mcs_boresight_meas'].to_dict('records')
    assert meas['pfs_visit_id'] == 153951
    assert meas['mcs_boresight_frame_set_id'] == 7
    assert meas['altitude'] == 60.0
    assert 0 < meas['mcs_boresight_x_err_pix'] < 0.1 and 0 < meas['mcs_boresight_y_err_pix'] < 0.1
    calculatedAt = pd.Timestamp(meas['calculated_at'])
    assert calculatedAt.tzinfo is not None
    assert abs((calculatedAt - pd.Timestamp(before)).total_seconds()) < 60


@pytest.mark.parametrize('insRots,error', [([-15, 15], 0.191), ([-45, 45], 0.070)])
def test_pairWrittenAtTheFloor(insRots, error):
    """A pair is written with the error FRAME_NOISE leads to for its separation, and a warning."""
    frames, points = loop(insRots)
    db, cmd = FakeDB(frames, points), Cmd()
    boresightMeasure.calcBoresight(cmd, db, frames.mcs_frame_id.tolist(), 153951, writeToDB=True)

    assert any('could not be checked' in text for text in cmd.warnings)
    [meas] = db.inserted['mcs_boresight_meas'].to_dict('records')
    assert meas['mcs_boresight_x_err_pix'] == pytest.approx(error, abs=0.001)


def test_narrowPairNotWritten():
    """Two frames 12 deg apart are measured and reported, but too uncertain to be written."""
    frames, points = loop([-6, 6])
    db, cmd = FakeDB(frames, points), Cmd()
    with pytest.raises(ValueError, match='too little rotation'):
        boresightMeasure.calcBoresight(cmd, db, frames.mcs_frame_id.tolist(), 153951, writeToDB=True)

    assert any(text.startswith('text="Boresight') for text in cmd.informs)
    assert db.inserted == {}


def test_displacedFrameNotWritten():
    """A loop with a frame still displaced once two were left out is reported, not written."""
    frames, points = loop(np.arange(-165, 166, 30))
    points.loc[points.mcs_frame_id.isin(frames.mcs_frame_id[[1, 5, 9]]), 'x'] += 3
    db, cmd = FakeDB(frames, points), Cmd()
    with pytest.raises(ValueError, match='still displaced'):
        boresightMeasure.calcBoresight(cmd, db, frames.mcs_frame_id.tolist(), 153951, writeToDB=True)

    assert any('still displaced' in text for text in cmd.warnings)
    assert db.inserted == {}


def test_notWrittenUnlessAsked():
    db = FakeDB(*loop([-15, 15, 45]))
    boresightMeasure.calcBoresight(Cmd(), db, [15395100, 15395101, 15395102], 153951, writeToDB=False)
    assert db.inserted == {}
