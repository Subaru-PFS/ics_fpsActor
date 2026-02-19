import os

import numpy as np
import pandas as pd
import pfs.utils.pfsDesignUtils as pfsDesignUtils
from pfs.datamodel import TargetType, PfsDesign
from pfs.utils.fiberids import FiberIds
from pfs.utils.pfsDesignUtils import fakeRaDecFromPfiNominal

pfsDesignDir = '/data/pfsDesign'
from pfs.utils import butler
from ics.fpsActor.utils.alfUtils import sgfm, dots


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


def createPfsDesign(calibModel, xy, moveTargetType, MOVE_MASK=None, designName=''):
    """Create a PfsDesign from xy positions, applying mask/targetType and faking ra/dec."""
    cobraMapping = sgfm.copy()
    MOVE_MASK = cobraMapping.COBRA_OK_MASK.to_numpy() if MOVE_MASK is None else np.asarray(MOVE_MASK, dtype=bool)

    if xy.shape[0] != len(cobraMapping) or xy.shape[1] != 2:
        raise ValueError(f'xy must have shape ({len(cobraMapping)}, 2), got {xy.shape}')

    if MOVE_MASK.shape[0] != len(cobraMapping):
        raise ValueError(f'MOVE_MASK must have length {len(cobraMapping)}, got {MOVE_MASK.shape[0]}')

    # setting positions.
    cobraMapping['x'] = xy[:, 0]
    cobraMapping['y'] = xy[:, 1]

    # setting targetType.
    cobraMapping['targetType'] = TargetType.UNASSIGNED
    cobraMapping.loc[MOVE_MASK, 'targetType'] = moveTargetType
    targetType = cobraMapping.sort_values('fiberId').targetType.to_numpy()

    # setting position to NaN where no target.
    cobraMapping.loc[~MOVE_MASK, 'x'] = np.nan
    cobraMapping.loc[~MOVE_MASK, 'y'] = np.nan

    # faking ra and dec.
    pfiNominal = cobraMapping.sort_values('fiberId')[['x', 'y']].to_numpy()
    ra, dec = fakeRaDecFromPfiNominal(pfiNominal)

    pfsDesign = pfsDesignUtils.makePfsDesign(pfiNominal=pfiNominal, ra=ra, dec=dec, targetType=targetType,
                                             arms='brnm', designName=designName)
    # Set BROKENFIBER, BROKENCOBRA, BLOCKED fiberStatus.
    pfsDesign = pfsDesignUtils.setFiberStatus(pfsDesign, calibModel=calibModel)

    return pfsDesign


def createHomeDesign(calibModel, positions, movingIdx, designName=''):
    """Create home design from current calibModel, ra and dec are faked."""
    xy = np.column_stack((np.real(positions), np.imag(positions)))
    MOVE_MASK = np.isin(sgfm.cobraId.to_numpy() - 1, movingIdx)

    return createPfsDesign(calibModel, xy, TargetType.HOME, MOVE_MASK=MOVE_MASK, designName=designName)


def createBlackDotDesign(calibModel, movingIdx, designName=''):
    """Create black dots design from current dots position, ra and dec are faked."""
    xy = np.column_stack((dots.x.to_numpy(), dots.y.to_numpy()))
    MOVE_MASK = np.isin(sgfm.cobraId.to_numpy() - 1, movingIdx)

    return createPfsDesign(calibModel, xy, TargetType.BLACKSPOT, MOVE_MASK=MOVE_MASK, designName=designName)


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
