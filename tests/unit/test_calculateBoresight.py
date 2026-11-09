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


def test_twoFrames(measured):
    """Two frames from different visits are recorded under the visit of the first."""
    cmd = run(frameIds=(15255101, 15255000), writeToDB=())
    assert measured == [([15255101, 15255000], 152551, True)]
    assert cmd.finished


@pytest.mark.parametrize('frameIds,match', [
    ((15255000,), 'needs two frames'),
    ((15255000, 15255001, 15255002), 'INSTRM-3100'),
])
def test_refused(measured, frameIds, match):
    """Anything but two frames is refused before measuring."""
    with pytest.raises(ValueError, match=match):
        run(frameIds=frameIds)
    assert measured == []
