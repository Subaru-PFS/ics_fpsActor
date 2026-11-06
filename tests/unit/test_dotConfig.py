"""fps dotConfig: the convergence values set until fps restarts, and the dotConfig keyword."""
import types

import opscore.protocols.types as opsTypes
import pytest

from ics.fpsActor.Commands.FpsCmd import FpsCmd

CONFIG = dict(convergence=dict(dotLanding=0.1, dotTarget=0.5, dotLandingAim=False, dotCalibration=True))


class Cmd:
    def __init__(self, **keywords):
        self.replies = []
        self.cmd = types.SimpleNamespace(keywords={k: types.SimpleNamespace(values=[v])
                                                   for k, v in keywords.items()})

    def inform(self, text):
        self.replies.append(('i', text))

    def finish(self, text=''):
        self.replies.append((':', text))

    def fail(self, text):
        self.replies.append(('f', text))

    def warn(self, text):
        self.replies.append(('w', text))

    def keyword(self):
        return [t for c, t in self.replies if t.startswith('dotConfig=')][-1]


@pytest.fixture
def fps():
    cmdSet = object.__new__(FpsCmd)
    cmdSet.actor = types.SimpleNamespace(actorConfig=CONFIG, dotOverride={})
    return cmdSet


def test_withoutAnOverrideTheConfigApplies(fps):
    cmd = Cmd()
    fps.dotConfig(cmd)
    assert cmd.keyword() == 'dotConfig=0.1,0.5,off,config' and cmd.replies[-1][0] == ':'


def test_anOverrideReplacesOnlyWhatItSets(fps):
    fps.dotConfig(Cmd(dotLanding=0.2, dotLandingAim=opsTypes.Bool('off', 'on')('on')))
    cmd = Cmd()
    fps.dotConfig(cmd)
    assert cmd.keyword() == 'dotConfig=0.2,0.5,on,override'
    assert fps._convergenceConfig()['dotCalibration'] is True
    assert fps.actor.dotOverride == dict(dotLanding=0.2, dotLandingAim=True)


def test_anImpossibleLandingIsRefusedAndTheValuesKept(fps):
    fps.dotConfig(Cmd(dotLanding=0.2))
    cmd = Cmd(dotLanding=0.6)                       # deeper than the target
    fps.dotConfig(cmd)
    assert cmd.replies[-1][0] == 'f' and fps.actor.dotOverride == dict(dotLanding=0.2)


def test_resetReturnsToTheConfig(fps):
    fps.dotConfig(Cmd(dotLanding=0.2))
    cmd = Cmd(reset=None)
    fps.dotConfig(cmd)
    assert cmd.keyword() == 'dotConfig=0.1,0.5,off,config' and not fps.actor.dotOverride
    cmd = Cmd(reset=None, dotLanding=0.2)
    fps.dotConfig(cmd)
    assert cmd.replies[-1][0] == 'f'


def test_aTargetPastTheDeepCapIsRefused(fps):
    cmd = Cmd(dotTarget=0.95)
    fps.dotConfig(cmd)
    assert cmd.replies[-1][0] == 'f' and not fps.actor.dotOverride


def test_anOverrideTheConfigNoLongerAdmitsIsWarnedOf(fps):
    fps.actor.dotOverride = dict(dotTarget=0.08)      # shallower than fps.yaml's landing
    cmd = Cmd()
    fps._informDotConfig(cmd)
    assert cmd.replies == [('w', cmd.replies[0][1])] and 'dotConfig' in cmd.replies[0][1]
