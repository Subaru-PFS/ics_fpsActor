"""Tests for the blind move, which is the one step nothing measures afterwards.

Everything here guards against commanding a large open-loop push on bad information:
a cobra that is still visible, an estimate that has diverged, or a run directory that
does not describe the cobras it claims to.
"""
import numpy as np
import pandas as pd
import pytest

from ics.cobraCharmer.cobraCoach.engineer import moveDtype
from ics.fpsActor.utils import dotGeometry, dotMove


def test_mismatchedRunDirectoryIsRejected(tmp_path):
    """Rows are positions in the commanded set; without matching ids they mean nothing."""
    data = tmp_path / 'data'
    data.mkdir()
    rows = np.zeros((4, 6), dtype=moveDtype)
    np.save(data / 'moves.npy', rows)
    np.savez(data / 'cobra_filtering.npz',
             final_moving_cobras=np.array([0, 1]),
             to_black_dot=np.zeros(4, dtype=bool))

    class Runner:
        dataDir = str(data)

    class Coach:
        runManager = Runner()

    with pytest.raises(ValueError):
        dotMove.loadRunInputs(Coach())


def test_threeDimensionalMovesAreAccepted(tmp_path):
    """moveToPfsDesign writes (1, nCobras, nIter) on the twoSteps path."""
    data = tmp_path / 'data'
    data.mkdir()
    np.save(data / 'moves.npy', np.zeros((1, 2, 6), dtype=moveDtype))
    np.savez(data / 'cobra_filtering.npz',
             final_moving_cobras=np.array([7, 9]),
             to_black_dot=np.array([False] * 7 + [True, False, True]))

    class Runner:
        dataDir = str(data)

    class Coach:
        runManager = Runner()

    rows, rowCobraId, dotIdx = dotMove.loadRunInputs(Coach())
    assert rows.shape == (2, 6)
    assert list(rowCobraId) == [7, 9]
    assert list(dotIdx) == [7, 9]


def test_plausibleRangeAdmitsACobraOutsideItsDot():
    """Never having entered the dot is a legitimate outcome, not a diverged estimate."""
    low, high = dotMove.PLAUSIBLE_FRACTION
    assert low < 0.0, 'a cobra short of the entry edge must still be movable'
    assert high > 1.0, 'a cobra past the exit edge must still be movable'


class TestLitFromFlux:
    """Selecting which cobras a hiding sequence should still push.

    Every branch here decides whether a cobra gets an open-loop move, so the failure is
    always the same one: pushing a cobra that nobody measured.
    """
    N = 10

    def frame(self, pairs):
        return pd.DataFrame(dict(cobra_id=[c for c, _ in pairs],
                                 flux_ratio_norm=[f for _, f in pairs]))

    def test_aboveThresholdIsLit(self):
        lit = dotMove.litFromFlux(self.frame([(1, 0.9), (2, 0.001)]), self.N)
        assert lit[0] and not lit[1]

    def test_thresholdItselfCountsAsHidden(self):
        """At the boundary the cobra has reached the depth asked of it."""
        lit = dotMove.litFromFlux(self.frame([(1, dotMove.HIDDEN_FLUX)]), self.N)
        assert not lit[0]

    def test_unmeasuredCobraIsNotPushed(self):
        """NaN means the flat could not measure it, which is not evidence of being lit."""
        lit = dotMove.litFromFlux(self.frame([(1, np.nan)]), self.N)
        assert not lit[0]

    def test_missingRowIsNotPushed(self):
        lit = dotMove.litFromFlux(self.frame([(1, 0.9)]), self.N)
        assert lit.sum() == 1

    def test_noRowsIsNotAHiddenFleet(self):
        """drp writing nothing must not read as every cobra being behind its dot."""
        assert dotMove.litFromFlux(self.frame([]), self.N) is None
        assert dotMove.litFromFlux(None, self.N) is None

    def test_rowsPresentButNoneLitIsSuccess(self):
        """Distinct from the case above: measured, and all of them hidden."""
        lit = dotMove.litFromFlux(self.frame([(1, 0.001), (2, 0.002)]), self.N)
        assert lit is not None and not lit.any()

    def test_cobraIdOutOfRangeIsIgnored(self):
        """Id zero would index the last cobra and push an unrelated one."""
        lit = dotMove.litFromFlux(self.frame([(0, 0.9), (self.N + 1, 0.9), (-3, 0.9)]),
                                  self.N)
        assert not lit.any()

    def test_thresholdIsWorthTheDepthItGivesUp(self):
        """Replayed on the three scans, 0.01 stops ~3x short of a cobra's own floor.

        Above ~0.02 that grows past ten times; below ~0.005 the cobras whose floor lies
        above the level are stepped every remaining flat and end worse than they started.
        """
        assert 0.005 <= dotMove.HIDDEN_FLUX <= 0.02


def test_dotDepthsTakeTheKeywordThenTheConfigThenTheDefaults():
    config = dict(dotLanding=0.15, dotTarget=0.45)

    assert dotMove.dotDepths(config, dotLanding=0.12, dotTarget=0.4) == (0.12, 0.4)
    assert dotMove.dotDepths(config) == (0.15, 0.45)
    assert dotMove.dotDepths(config, dotTarget=0.6) == (0.15, 0.6)
    assert dotMove.dotDepths({}) == (dotGeometry.RAMP_LANDING_FRACTION,
                                     dotGeometry.BLIND_TARGET_FRACTION)


@pytest.mark.parametrize('landing, target', [(0.1, 0.1), (0.3, 0.2), (0.0, 0.5), (0.1, 1.0),
                                             (0.1, 5.0), (-0.1, 0.5)])
def test_dotDepthsRefuseAPushThatDoesNotGoDeeper(landing, target):
    with pytest.raises(ValueError, match='dotLanding'):
        dotMove.dotDepths({}, dotLanding=landing, dotTarget=target)


def test_dotDepthsRefuseAMissingConfigValue():
    with pytest.raises(TypeError):
        dotMove.dotDepths(dict(dotTarget=None))


def test_dotDepthsTargetTheLandingWithoutABlindMove():
    config = dict(dotLanding=0.1, dotTarget=0.5)

    assert dotMove.dotDepths(config, dotLanding=0.6, noBlindMove=True) == (0.6, 0.6)
    assert dotMove.dotDepths(config, dotTarget=0.4, noBlindMove=True) == (0.1, 0.1)
    with pytest.raises(ValueError, match='dotLanding'):
        dotMove.dotDepths(config, dotLanding=1.0, noBlindMove=True)


def test_dotApproachTakesTheConfigThenTheDefaults():
    assert dotMove.dotApproach({}) == (dotGeometry.RAMP_APPROACH_FRACTION, 0.05)
    assert dotMove.dotApproach(dict(dotApproach=-0.1, dotApproachTolerance=0.03)) == (-0.1, 0.03)
    assert dotMove.dotApproach(dict(dotApproach=dotGeometry.APPROACH_RANGE[0]))[0] == dotGeometry.APPROACH_RANGE[0]


@pytest.mark.parametrize('config', [dict(dotApproach=0.0), dict(dotApproach=-0.03), dict(dotApproach=-0.6),
                                    dict(dotApproach=0.05), dict(dotApproachTolerance=0.0),
                                    dict(dotApproachTolerance=-0.025)])
def test_dotApproachRefusesAnApproachOutsideItsRangeOrNoTolerance(config):
    with pytest.raises(ValueError):
        dotMove.dotApproach(config)


def test_landingFractionsAimFromTheApproach():
    """A calibrated cobra with landing gain 2 travels twice its command from the approach."""
    gains = pd.DataFrame(dict(landing_gain=[2.0], target_gain=[1.0], calibrated=[True]),
                         index=pd.Index([1], name='cobra_id'))
    for approach in (-0.075, -0.1):
        command, aimed = dotMove.landingFractions(0.2, gains, 1, approach=approach)
        assert aimed[0] and command[0] == pytest.approx(approach + (0.2 - approach) / 2)
