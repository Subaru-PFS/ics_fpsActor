"""Which frames fps calculateBoresight measures, from its keywords."""
import types

import pytest

from ics.fpsActor.Commands import FpsCmd


class Keyword:
    def __init__(self, *values):
        self.values = list(values)


class Cmd:
    """Records how the command finished."""

    def __init__(self, **keywords):
        self.cmd = types.SimpleNamespace(keywords={name: Keyword(*values) for name, values in keywords.items()})
        self.finished = None

    def inform(self, text):
        pass

    def finish(self, text):
        self.finished = text


@pytest.fixture
def measured(monkeypatch):
    """The (frameIds, pfsVisitId, writeToDB) calcBoresight is called with, if it is."""
    calls = []
    monkeypatch.setattr(FpsCmd.boresightMeasure, 'calcBoresight',
                        lambda cmd, db, frameIds, pfsVisitId, writeToDB: calls.append(
                            (list(frameIds), pfsVisitId, writeToDB)))
    return calls


def run(**keywords):
    cmd = Cmd(**keywords)
    FpsCmd.FpsCmd.calculateBoresight(types.SimpleNamespace(connectToDB=lambda cmd: None), cmd)
    return cmd


def test_framesAcrossVisits(measured):
    """Frames from different visits are recorded under the visit of the first."""
    cmd = run(frameIds=(15255101, 15255000), writeToDB=())
    assert measured == [([15255101, 15255000], 152551, True)]
    assert cmd.finished


def test_loop(measured):
    """A whole boresight loop is measured at once."""
    frameIds = tuple(15395100 + i for i in range(12))
    cmd = run(frameIds=frameIds)
    assert measured == [(list(frameIds), 153951, False)]
    assert cmd.finished


def test_oneFrameRefused(measured):
    """A single frame is refused before measuring."""
    with pytest.raises(ValueError, match='needs two frames'):
        run(frameIds=(15255000,))
    assert measured == []
