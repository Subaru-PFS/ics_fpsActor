"""Tests for the dot-target read path.

The failure that matters is not an exception -- it is a plausible number.  Every cobra
has a row so the file is a complete table, but only some rows carry a measurement, and
returning the filler for the rest would quietly present the fleet default as if it had
been measured.  Most of what follows guards that boundary.
"""
import numpy as np
import pytest

from ics.fpsActor.utils import dotTargets

N_COBRAS = 10

HEADER = 'cobraId,dotTargetFraction,calibrated,dotEdgeFraction,entryEdgeSeen,runSpread\n'


def writeCsv(tmp_path, rows, header=HEADER):
    path = tmp_path / 'targets.csv'
    path.write_text(header + ''.join(rows))
    return str(path)


class FakeCmd:
    def __init__(self):
        self.informs, self.warns = [], []

    def inform(self, text):
        self.informs.append(text)

    def warn(self, text):
        self.warns.append(text)


def test_missingFileGivesNoMeasurements():
    cmd = FakeCmd()
    targets = dotTargets.loadDotTargets(N_COBRAS, path='/no/such/file.csv', cmd=cmd)
    assert targets.shape == (N_COBRAS,)
    assert np.isnan(targets).all()
    assert len(cmd.warns) == 1


def test_unreadableFileGivesNoMeasurements(tmp_path):
    """A directory where a file is expected raises something other than FileNotFoundError."""
    cmd = FakeCmd()
    targets = dotTargets.loadDotTargets(N_COBRAS, path=str(tmp_path), cmd=cmd)
    assert np.isnan(targets).all()
    assert len(cmd.warns) == 1


def test_headerOnlyGivesNoMeasurements(tmp_path):
    targets = dotTargets.loadDotTargets(N_COBRAS, path=writeCsv(tmp_path, []))
    assert np.isnan(targets).all()


def test_fillerIsNotReturnedAsMeasured(tmp_path):
    """An uncalibrated row carries the fleet default so every cobra has a row.

    Returning it would present that default as a measurement, and the caller would have
    no way to tell the two apart.
    """
    path = writeCsv(tmp_path, ['1,0.4207,True,0.036,False,0.06\n',
                               '2,0.4450,False,,False,\n',
                               '3,0.5211,True,0.123,True,0.01\n'])
    targets = dotTargets.loadDotTargets(N_COBRAS, path=path)
    assert targets[0] == pytest.approx(0.4207)
    assert np.isnan(targets[1])
    assert targets[2] == pytest.approx(0.5211)


@pytest.mark.parametrize('flag,measured', [('True', True), ('true', True), ('1', True),
                                           ('yes', True), ('False', False),
                                           ('false', False), ('0', False), ('', False)])
def test_calibratedFlagParsing(tmp_path, flag, measured):
    """An unset flag means not calibrated -- absence of evidence, not evidence of one."""
    path = writeCsv(tmp_path, [f'1,0.5000,{flag},0.1,True,0.01\n'])
    targets = dotTargets.loadDotTargets(N_COBRAS, path=path)
    assert np.isfinite(targets[0]) == measured


def test_oneBadRowDoesNotDiscardTheFile(tmp_path):
    path = writeCsv(tmp_path, ['1,0.4207,True,0.03,False,0.06\n',
                               '2,abc,True,0.03,False,0.06\n',
                               '3,,True,0.03,False,0.06\n',
                               '4,0.5211,True,0.12,True,0.01\n'])
    targets = dotTargets.loadDotTargets(N_COBRAS, path=path)
    assert targets[0] == pytest.approx(0.4207)
    assert np.isnan(targets[1])
    assert np.isnan(targets[2])
    assert targets[3] == pytest.approx(0.5211)


def test_cobraIdOutOfRangeIsIgnored(tmp_path):
    """Id zero is the dangerous one: id - 1 would index the last cobra."""
    path = writeCsv(tmp_path, ['0,0.5000,True,0.1,True,0.01\n',
                               '-1,0.5000,True,0.1,True,0.01\n',
                               f'{N_COBRAS + 1},0.5000,True,0.1,True,0.01\n'])
    targets = dotTargets.loadDotTargets(N_COBRAS, path=path)
    assert np.isnan(targets).all()


def test_extrapolatedTargetsAreRejected(tmp_path):
    """Outside the scanned range the fit is extrapolating the profile, not measuring it.

    Clamping to the boundary would still assert a depth nobody observed.
    """
    path = writeCsv(tmp_path, ['1,0.0120,True,0.0,True,0.005\n',
                               '2,0.9271,True,0.0,True,0.053\n',
                               '3,0.5000,True,0.1,True,0.010\n'])
    targets = dotTargets.loadDotTargets(N_COBRAS, path=path)
    assert np.isnan(targets[0])
    assert np.isnan(targets[1])
    assert targets[2] == pytest.approx(0.5)


def test_returnLengthFollowsTheRequest(tmp_path):
    """A partial bench runs fewer cobras than the file describes."""
    path = writeCsv(tmp_path, [f'{i},0.5,True,0.1,True,0.01\n' for i in range(1, 9)])
    assert dotTargets.loadDotTargets(4, path=path).shape == (4,)


def test_provenanceNamesTheCalibratedCount(tmp_path):
    """The operator has to be able to see that a fallback happened."""
    cmd = FakeCmd()
    path = writeCsv(tmp_path, ['1,0.4207,True,0.03,False,0.06\n',
                               '2,0.4450,False,,False,\n'])
    dotTargets.loadDotTargets(N_COBRAS, path=path, cmd=cmd)
    assert len(cmd.informs) == 1
    assert 'targets.csv' in cmd.informs[0]
    assert '1 of 10' in cmd.informs[0]


def test_resolveFillsTheGapsWithTheDefault(tmp_path):
    path = writeCsv(tmp_path, ['1,0.4207,True,0.03,False,0.06\n',
                               '2,0.4450,False,,False,\n'])
    fractions, measured = dotTargets.resolveTargets(N_COBRAS, 0.445, path=path)
    assert fractions[0] == pytest.approx(0.4207)
    assert fractions[1] == pytest.approx(0.445)
    assert measured[0] and not measured[1]
    assert np.isfinite(fractions).all()


def test_resolveSurvivesAMissingFile():
    """A convergence must not fail because a calibration is absent."""
    fractions, measured = dotTargets.resolveTargets(N_COBRAS, 0.445, path='/no/file.csv')
    assert fractions == pytest.approx(np.full(N_COBRAS, 0.445))
    assert not measured.any()


def test_defaultPathComesFromTheButler(monkeypatch):
    """With no path given, the file is the cobraDotTarget product of pfs_instdata."""
    try:
        path = dotTargets.Butler().getPath(dotTargets.PRODUCT)
    except KeyError:
        pytest.skip('installed pfs_utils has no cobraDotTarget product')
    assert str(path).endswith('pfi/dot/cobra_dot_target.csv')
    targets = dotTargets.loadDotTargets(2394)
    measured = np.isfinite(targets)
    if not measured.any():
        return                       # product set up without the file: uncalibrated, not an error
    assert np.all(targets[measured] >= dotTargets.TARGET_BOUNDS[0])
    assert np.all(targets[measured] <= dotTargets.TARGET_BOUNDS[1])
    assert abs(np.median(targets[measured]) - 0.50) < 0.03


def test_commentLinesAreNotRows(tmp_path):
    """A provenance header in the product must not be read as a cobra."""
    path = writeCsv(tmp_path, ['# fitted to the dot scan of 2026-09-01\n',
                               '1,0.4207,True,,False,\n'])
    targets = dotTargets.loadDotTargets(N_COBRAS, path=path)
    assert targets[0] == pytest.approx(0.4207)
    assert np.isnan(targets[1:]).all()
