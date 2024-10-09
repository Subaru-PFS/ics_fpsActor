from datetime import datetime as datetime

import numpy as np
import pandas as pd
import pfs.utils.coordinates.updateTargetPosition as updateTargetPosition
import pfs.utils.ingestPfsDesign as ingestPfsDesign
import pfs.utils.pfsConfigUtils as pfsConfigUtils
import pfs.utils.pfsDesignUtils as pfsDesignUtils
from ics.fpsActor.utils.pfsDesign import readDesign
from opdb import opdb
from pfs.datamodel import PfsConfig, FiberStatus, TargetType
from pfs.utils.fiberids import FiberIds
from scipy.interpolate import griddata

__all__ = ["pfsConfigFromDesign", "makeVanillaPfsConfig", "makeTargetsArray", "tweakTargetPosition",
           "updatePfiCenter", "writePfsConfig", "ingestPfsConfig"]


def pfsConfigFromDesign(pfsDesign, visit0):
    """Just make a PfsConfig file identical to PfsDesign."""
    return PfsConfig.fromPfsDesign(pfsDesign=pfsDesign, visit=visit0, pfiCenter=pfsDesign.pfiNominal)


def makeVanillaPfsConfig(pfsDesignId, visit0, maskFile=None):
    """Load pfsDesign and return a PfsConfig file identical to PfsDesign."""
    pfsDesign = readDesign(pfsDesignId)

    if maskFile:
        # retrieving masked cobras/fibers.
        maskFile = pd.read_csv(maskFile, index_col=0)
        noTarget = maskFile[maskFile.bitMask == 0]
        noEng = pfsDesign.targetType != TargetType.ENGINEERING
        # setting non-engineering fiber which are masked to UNASSIGNED.
        noTargetMask = np.logical_and(noEng, np.isin(pfsDesign.fiberId, noTarget.fiberId))
        pfsDesign.targetType[noTargetMask] = TargetType.UNASSIGNED

    return pfsConfigFromDesign(pfsDesign, visit0)


def makeTargetsArray(pfsConfig):
    """Construct target array from pfsConfig file."""
    allCobraIds = np.arange(2394, dtype='int32') + 1
    fiberId = pfsConfig.fiberId
    cobraId = FiberIds().fiberIdToCobraId(fiberId)
    # targets vector has an entry for each cobra and sorted by cobraId.
    targets = np.empty((2394, 2), dtype=pfsConfig.pfiNominal.dtype)
    targets[:] = np.NaN
    # cobraMask is boolean array(shape=cobraId.shape)
    cobraMask = np.isin(cobraId, allCobraIds)
    # only existing cobraId.
    cobraId = cobraId[cobraMask]
    # assigning target vector directly.
    targets[cobraId - 1] = pfsConfig.pfiNominal[cobraMask]
    isNan = np.logical_or(np.isnan(targets[:, 0]), np.isnan(targets[:, 1]))

    return targets[:, 0] + targets[:, 1] * 1j, isNan


def tweakTargetPosition(pfsConfig, cmd=None):
    """Update pfsConfig target position at the time of observation."""
    radec = np.vstack([pfsConfig.ra, pfsConfig.dec])
    pa = pfsConfig.posAng
    cent = np.vstack([pfsConfig.raBoresight, pfsConfig.decBoresight])
    # getting pm and par from design.
    pm = np.vstack([pfsConfig.pmRa, pfsConfig.pmDec])
    par = pfsConfig.parallax
    obstime = datetime.utcnow().isoformat()

    ra_now, dec_now, pfi_now_x, pfi_now_y = updateTargetPosition.update_target_position(radec, pa, cent, pm, par,
                                                                                        obstime)
    # setting the new positions.
    pfsConfig.ra = ra_now
    pfsConfig.dec = dec_now
    pfsConfig.pfiNominal = np.vstack((pfi_now_x, pfi_now_y)).transpose()

    return pfsConfig


def updatePfiCenter(pfsConfig, calibModel, cmd=None, noMatchStatus=FiberStatus.BLACKSPOT,
                    notConvergedDistanceThreshold=None):
    """Update final cobra positions after converging to pfsDesign."""

    def fetchFinalConvergence(visitId):
        """Retrieve final cobra position in mm.

        Parameters
        ----------
        visitId : `int`
            Convergence identifier.
        """
        sql = 'SELECT pfs_visit_id, iteration, cobra_id, cobra_match.spot_id, pfi_center_x_mm, pfi_center_y_mm ' \
              'FROM mcs_data LEFT OUTER JOIN cobra_match ON mcs_data.spot_id = cobra_match.spot_id AND mcs_data.mcs_frame_id = cobra_match.mcs_frame_id ' \
              f'WHERE cobra_match.pfs_visit_id={visitId} AND iteration=(select max(cm2.iteration) from cobra_match cm2 WHERE cm2.pfs_visit_id = {visitId}) ' \
              'order by cobra_id asc'

        db = opdb.OpDB(hostname="db-ics", username="pfs", dbname="opdb")
        lastIteration = db.fetch_query(sql)
        return lastIteration

    # Retrieve dataset
    lastIteration = fetchFinalConvergence(pfsConfig.visit)

    # Setting missing matches to NaNs.
    NO_MATCH_MASK = lastIteration.spot_id == -1
    lastIteration.loc[NO_MATCH_MASK, 'pfi_center_x_mm'] = np.NaN
    lastIteration.loc[NO_MATCH_MASK, 'pfi_center_y_mm'] = np.NaN

    # Fill final position with NaNs.
    pfiCenter = np.empty(pfsConfig.pfiNominal.shape, dtype=pfsConfig.pfiNominal.dtype)
    pfiCenter[:] = np.NaN

    # Construct the index.
    fiberId = FiberIds().cobraIdToFiberId(lastIteration.cobra_id.to_numpy())
    lastIteration['fiberId'] = fiberId
    fiberIndex = pd.DataFrame(dict(fiberId=pfsConfig.fiberId, tindex=np.arange(len(pfsConfig.fiberId))))
    fiberIndex = fiberIndex.set_index('fiberId').loc[fiberId].tindex.to_numpy()

    # Set final cobra position.
    pfiCenter[fiberIndex, 0] = lastIteration.pfi_center_x_mm.to_numpy()
    pfiCenter[fiberIndex, 1] = lastIteration.pfi_center_y_mm.to_numpy()
    pfsConfig.pfiCenter = pfiCenter

    # Calculate distance to target.
    distanceToTarget = np.hypot(pfsConfig.pfiNominal[:, 0] - pfsConfig.pfiCenter[:, 0],
                                pfsConfig.pfiNominal[:, 1] - pfsConfig.pfiCenter[:, 1])

    # Set BROKENFIBER, BROKENCOBRA, BLOCKED fiberStatus.
    pfsConfig = pfsDesignUtils.setFiberStatus(pfsConfig, calibModel=calibModel)

    # Populating the dataframe for convenience.
    lastIteration['fiberStatus'] = pfsConfig.fiberStatus[fiberIndex]
    lastIteration['targetType'] = pfsConfig.targetType[fiberIndex]
    lastIteration['ra'] = pfsConfig.ra[fiberIndex]
    lastIteration['dec'] = pfsConfig.dec[fiberIndex]
    lastIteration[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']] = pfsConfig.pfiNominal[fiberIndex]
    lastIteration['distanceToTarget'] = distanceToTarget[fiberIndex]

    # Making fiberStatus, targetType masks.
    FIBER_GOOD_MASK = lastIteration.fiberStatus.to_numpy() == FiberStatus.GOOD
    WITH_TARGET_MASK = lastIteration.targetType.isin([TargetType.SCIENCE, TargetType.SKY, TargetType.FLUXSTD])
    UNASSIGNED_TARGET_MASK = lastIteration.targetType == TargetType.UNASSIGNED

    # Set fiberStatus for the not matched cobras, BLACKSPOT or NOTCONVERGED.
    lastIteration.loc[FIBER_GOOD_MASK & NO_MATCH_MASK, 'fiberStatus'] = noMatchStatus

    # setting NOTCONVERGED fiberStatus for cobra above distance threshold.
    if notConvergedDistanceThreshold:
        aboveThreshold = lastIteration.distanceToTarget.to_numpy() > notConvergedDistanceThreshold
        cobraMask = aboveThreshold & WITH_TARGET_MASK & FIBER_GOOD_MASK & ~NO_MATCH_MASK
        lastIteration.loc[cobraMask, 'fiberStatus'] = FiberStatus.NOTCONVERGED

    pfsConfig.fiberStatus[fiberIndex] = lastIteration.fiberStatus.to_numpy()

    # Setting ra,dec for UNASSIGNED target.
    try:
        radec = griddata(lastIteration.loc[WITH_TARGET_MASK][['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].to_numpy(),
                         lastIteration.loc[WITH_TARGET_MASK][['ra', 'dec']].to_numpy(),
                         lastIteration.loc[UNASSIGNED_TARGET_MASK][['pfi_center_x_mm', 'pfi_center_y_mm']].to_numpy(),
                         method='cubic')
    except ValueError:
        radec = lastIteration.loc[UNASSIGNED_TARGET_MASK][['ra', 'dec']].to_numpy()  # return the same values.

    lastIteration.loc[UNASSIGNED_TARGET_MASK, ['ra', 'dec']] = radec
    pfsConfig.ra[fiberIndex] = lastIteration.ra.to_numpy()
    pfsConfig.dec[fiberIndex] = lastIteration.dec.to_numpy()

    if cmd:
        cmd.inform('text="pfsConfig updated successfully."')

    return lastIteration.iteration.max()


def writePfsConfig(pfsConfig, cmd=None):
    """Write final pfsConfig."""
    ret = pfsConfigUtils.writePfsConfig(pfsConfig)

    if cmd:
        cmd.inform(f'text="{pfsConfig.filename} written to disk')

    return ret


def ingestPfsConfig(pfsConfig, cmd=None, **kwargs):
    """Ingest PfsConfig file in opdb tables."""
    ret = ingestPfsDesign.ingestPfsConfig(pfsConfig, **kwargs)

    if cmd:
        cmd.inform('text="pfsConfig successfully inserted in opdb !"')

    return ret
