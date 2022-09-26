import numpy as np
import pandas as pd
from opdb import opdb
from pfs.datamodel import PfsDesign, PfsConfig
from pfs.utils.fiberids import FiberIds

pfsDesignDir = '/data/pfsDesign'
pfsConfigDir = '/data/drp/pfsDesign'


def fetchFinalConvergence(visitId):
    """Retrieve final cobra position in mm.

    Parameters
    ----------
    visitId : `int`
        Convergence identifier.
    """

    sql = f'select pfs_visit_id, iteration, cobra_id, pfi_center_x_mm, pfi_center_y_mm from cobra_match cm where ' \
          f'cm.iteration=(select max(cm2.iteration) from cobra_match cm2 where cm2.pfs_visit_id = {visitId}) ' \
          f'and cm.pfs_visit_id={visitId} order by cobra_id asc'

    db = opdb.OpDB(hostname="db-ics", username="pfs", dbname="opdb")
    lastIteration = db.fetch_query(sql)
    return lastIteration


def cobraIdToFiberId(cobraIds):
    """Return fiberIds for the specified cobraIds.

    Parameters
    ----------
    cobraIds : `numpy.ndarray` of `int`
        Array of 1-indexed cobraIds.
    """
    gfm = pd.DataFrame(FiberIds().data)
    fiberIds = gfm.set_index('cobraId').loc[cobraIds].fiberId.to_numpy()
    return fiberIds


def writePfsConfig(pfsDesignId, visitId):
    """Write pfsConfig with final cobra positions after converging to pfsDesign.

    Parameters
    ----------
    pfsDesignId : `int`
        PFI design identifier, specifies the intended top-end configuration.
    visitId : `int`
        Convergence identifier.
    """
    # Retrieve dataset
    pfsDesign = PfsDesign.read(pfsDesignId=pfsDesignId, dirName=pfsDesignDir)
    lastIteration = fetchFinalConvergence(visitId)
    # Fill final position with NaNs.
    pfiCenter = np.empty(pfsDesign.pfiNominal.shape, dtype=pfsDesign.pfiNominal.dtype)
    pfiCenter[:] = np.NaN
    # Construct the index.
    fiberId = cobraIdToFiberId(lastIteration.cobra_id.to_numpy())
    fiberIndex = pd.DataFrame(dict(fiberId=pfsDesign.fiberId, tindex=np.arange(len(pfsDesign.fiberId))))
    fiberIndex = fiberIndex.set_index('fiberId').loc[fiberId].tindex.to_numpy()
    # Set final cobra position.
    pfiCenter[fiberIndex, 0] = lastIteration.pfi_center_x_mm.to_numpy()
    pfiCenter[fiberIndex, 1] = lastIteration.pfi_center_y_mm.to_numpy()
    # instantiate and write pfsConfig to disk.
    pfsConfig = PfsConfig.fromPfsDesign(pfsDesign=pfsDesign, visit0=visitId, pfiCenter=pfiCenter)
    pfsConfig.write(dirName=pfsConfigDir)

    return pfsConfig