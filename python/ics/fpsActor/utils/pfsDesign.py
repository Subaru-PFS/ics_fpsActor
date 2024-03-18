import os

import numpy as np
import pandas as pd
import pfs.utils.pfsDesignUtils as pfsDesignUtils
from pfs.datamodel import TargetType, PfsDesign, FiberStatus
from pfs.utils.fiberids import FiberIds
from pfs.utils.pfsDesignUtils import fakeRa, fakeDec, fakeRaDecFromPfiNominal

pfsDesignDir = '/data/pfsDesign'
from pfs.utils import butler


def makeDesignName(flavour, maskFile):
    """construct pfsDesign name."""
    if not maskFile:
        return flavour

    _, maskFileName = os.path.split(os.path.splitext(maskFile)[0])
    return f'{flavour}-{maskFileName}'


def readDesign(pfsDesignId):
    """Read PfsDesign from pfsDesignDir."""
    return PfsDesign.read(pfsDesignId=pfsDesignId, dirName=pfsDesignDir)


def writeDesign(pfsDesign):
    """Write PfsDesign to pfsDesignDir, do not override."""
    fullPath = os.path.join(pfsDesignDir, pfsDesign.filename)
    doWrite = not os.path.isfile(fullPath)

    if doWrite:
        pfsDesign.write(pfsDesignDir)

    return doWrite, fullPath


def createHomeDesign(calibModel, goodIdx, maskFile):
    """Create home design from current calibModel, ra and dec are faked."""
    gfm = pd.DataFrame(FiberIds().data)
    sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')
    sgfm['x'] = np.real(calibModel.centers)
    sgfm['y'] = np.imag(calibModel.centers)

    # setting targetType.
    MOVE_MASK = np.isin(sgfm.cobraId - 1, goodIdx)
    sgfm['targetType'] = TargetType.UNASSIGNED
    sgfm.loc[MOVE_MASK, 'targetType'] = TargetType.HOME
    targetType = sgfm.sort_values('fiberId').targetType.to_numpy()

    # setting position to NaN where no target.
    sgfm.loc[~MOVE_MASK, 'x'] = np.NaN
    sgfm.loc[~MOVE_MASK, 'y'] = np.NaN

    # faking ra and dec.
    pfiNominal = sgfm.sort_values('fiberId')[['x', 'y']].to_numpy()
    ra, dec = fakeRaDecFromPfiNominal(pfiNominal)

    # Set fiberStatus.
    sgfm['fiberStatus'] = FiberStatus.GOOD
    FIBER_BROKEN_MASK = (calibModel.status & calibModel.FIBER_BROKEN_MASK).astype('bool')
    sgfm.loc[FIBER_BROKEN_MASK, 'fiberStatus'] = FiberStatus.BROKENFIBER
    fiberStatus = sgfm.sort_values('fiberId')['fiberStatus'].to_numpy()

    # setting designName.
    designName = makeDesignName('cobraHome', maskFile)

    pfsDesign = pfsDesignUtils.makePfsDesign(pfiNominal=pfiNominal, ra=ra, dec=dec, targetType=targetType,
                                             fiberStatus=fiberStatus, arms='brn', designName=designName)

    return pfsDesign


def createBlackDotDesign(calibModel, goodIdx, maskFile):
    """Create blac dots design from current dots position, ra and dec are faked."""
    nestor = butler.Butler()
    dots = nestor.get('black_dots')

    gfm = pd.DataFrame(FiberIds().data)
    sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')
    sgfm['x'] = dots.x.to_numpy()
    sgfm['y'] = dots.y.to_numpy()

    # setting targetType.
    MOVE_MASK = np.isin(sgfm.cobraId - 1, goodIdx)
    sgfm['targetType'] = TargetType.UNASSIGNED
    sgfm.loc[MOVE_MASK, 'targetType'] = TargetType.BLACKSPOT
    targetType = sgfm.sort_values('fiberId').targetType.to_numpy()

    # setting position to NaN where no target.
    sgfm.loc[~MOVE_MASK, 'x'] = np.NaN
    sgfm.loc[~MOVE_MASK, 'y'] = np.NaN

    # faking ra and dec.
    pfiNominal = sgfm.sort_values('fiberId')[['x', 'y']].to_numpy()
    ra, dec = fakeRaDecFromPfiNominal(pfiNominal)

    # Set fiberStatus.
    sgfm['fiberStatus'] = FiberStatus.GOOD
    FIBER_BROKEN_MASK = (calibModel.status & calibModel.FIBER_BROKEN_MASK).astype('bool')
    sgfm.loc[FIBER_BROKEN_MASK, 'fiberStatus'] = FiberStatus.BROKENFIBER
    fiberStatus = sgfm.sort_values('fiberId')['fiberStatus'].to_numpy()

    # setting designName.
    designName = makeDesignName('blackDots', maskFile)

    pfsDesign = pfsDesignUtils.makePfsDesign(pfiNominal=pfiNominal, ra=ra, dec=dec, targetType=targetType,
                                             fiberStatus=fiberStatus, arms='brn', designName=designName)

    return pfsDesign


def homeMaskFromDesign(pfsDesign):
    """Return cobra mask where targetType==HOME."""
    return cobraIndexFromDesign(pfsDesign, targetType=TargetType.HOME)


def cobraIndexFromDesign(pfsDesign, targetType):
    """Return cobra mask from a given pfsDesign and targetType."""
    gfm = pd.DataFrame(FiberIds().data)
    sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')

    fiberId = pfsDesign[pfsDesign.targetType == targetType].fiberId
    cobraIds = sgfm[sgfm.fiberId.isin(fiberId)].cobraId.to_numpy()
    return cobraIds - 1
