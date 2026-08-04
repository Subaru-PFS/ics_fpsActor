import logging

import numpy as np
import pandas as pd
import pfs.utils.ingestPfsDesign as ingestPfsDesign
import pfs.utils.pfsConfigUtils as pfsConfigUtils
import pfs.utils.pfsDesignUtils as pfsDesignUtils
from ics.cobraCharmer import targetValidation
from pfs.datamodel import PfsConfig, FiberStatus, TargetType, InstrumentStatusFlag
from pfs.utils.database import opdb
from pfs.utils.fiberids import FiberIds
from scipy.interpolate import griddata

__all__ = ["pfsConfigFromDesign", "makeTargetsArray", "finalize", "writePfsConfig",
           "ingestPfsConfig"]


def pfsConfigFromDesign(pfsDesign, visit0, calibModel=None, header=None, maskFile=None, versions=None):
    """Just make a PfsConfig file identical to PfsDesign."""
    if maskFile:
        # retrieving masked cobras/fibers.
        maskFile = pd.read_csv(maskFile, index_col=0)
        noTarget = maskFile[maskFile.bitMask == 0]
        noEng = pfsDesign.targetType != TargetType.ENGINEERING
        # setting non-engineering fiber which are masked to UNASSIGNED.
        noTargetMask = np.logical_and(noEng, np.isin(pfsDesign.fiberId, noTarget.fiberId))
        pfsDesign.targetType[noTargetMask] = TargetType.UNASSIGNED

    # Create pfsConfig with pfiCenter as nan.
    pfiCenter = np.full(pfsDesign.pfiNominal.shape, np.nan, dtype=pfsDesign.pfiNominal.dtype)
    pfsConfig = PfsConfig.fromPfsDesign(pfsDesign, visit0, pfiCenter, header=header, visit0=visit0, versions0=versions)
    # Set BROKENFIBER, BROKENCOBRA, BLOCKED fiberStatus.
    return pfsDesignUtils.setFiberStatus(pfsConfig, calibModel=calibModel)


N_COBRAS = 2394

# Types that carry a real sky target and must converge.  These are the ones an
# operator cares about losing.
SCIENCE_TARGET_TYPES = [TargetType.SCIENCE, TargetType.SKY, TargetType.FLUXSTD]

# Target types a cobra may carry in a convergence design: the science ones, plus the two
# that are parked rather than converged.  Anything else -- DCB, AFL, SUNSS, HOME,
# SCIENCE_MASKED -- means the design was built for another purpose.  ENGINEERING is
# absent on purpose: those fibers have no cobra, so they never appear on the 2394-wide
# grid these masks live on.
ACCEPTED_TARGET_TYPES = SCIENCE_TARGET_TYPES + [TargetType.UNASSIGNED, TargetType.BLACKSPOT]


def cobraRows(pfsConfig):
    """Rows of pfsConfig that drive a cobra, and the 0-based cobra index of each.

    Reads pfsConfig.cobraId, which pfsDesignUtils.setFiberStatus fills in; fibers with no
    cobra carry a CobraId sentinel outside 1..nCobras and drop out here.  That is the same
    partition FiberIds().fiberIdToCobraId gives -- the two disagree only on which sentinel
    they use for the 64 engineering fibers -- but it costs no fiber-map load.
    """
    isCobra = np.isin(pfsConfig.cobraId, np.arange(1, N_COBRAS + 1))

    return isCobra, pfsConfig.cobraId[isCobra] - 1


def makeTargetsArray(pfsConfig):
    """Design positions as a complex (nCobras,) array indexed by cobra, NaN where none."""
    isCobra, index = cobraRows(pfsConfig)
    targets = np.full((N_COBRAS, 2), np.nan, dtype=pfsConfig.pfiNominal.dtype)
    targets[index] = pfsConfig.pfiNominal[isCobra]

    return targets[:, 0] + targets[:, 1] * 1j


def getCobraTargetMask(pfsConfig, targetTypes):
    """Bool (nCobras,) — True for cobras whose targetType is in targetTypes."""
    isCobra, index = cobraRows(pfsConfig)
    mask = np.zeros(N_COBRAS, dtype=bool)
    mask[index] = np.isin(pfsConfig.targetType[isCobra], targetTypes)

    return mask


def summariseByTargetType(pfsConfig, flags, converging, toBlackDot, isBroken, isMasked):
    """What the convergence will do to each target type the design carries, as text.

    One line per target type present, then a total.  Every row reconciles --
    nDesign = converging + toBlackDot + broken + masked -- and the rows sum to the
    total, so a cobra belonging to no outcome shows up as arithmetic that does not add.

    Reasons are counted over the non-broken cobras only.  A broken cobra is a parked
    monitoring fiber and always carries a non-finite target, which otherwise doubles
    the count that matters.

    Parameters
    ----------
    pfsConfig : PfsConfig
        Design being converged to.
    flags : ndarray of uint16, (nCobras,)
        Per-cobra verdict from targetValidation.validateTargets.
    converging, toBlackDot, isBroken, isMasked : ndarray of bool, (nCobras,)
        The four outcomes, mutually exclusive and covering every cobra.

    Returns
    -------
    list of str
        Header, one row per target type, then a total row.
    """
    informational = targetValidation.FLAG_NAMES[targetValidation.IN_OVERLAPPING_REGION]

    def row(label, inRow):
        counts = targetValidation.summarise(flags, isScience=inRow & ~isBroken)
        reasons = ', '.join(f'{name}={n}' for name, n in counts.items()
                            if n and name not in ('checked', 'invalid', informational))
        return (f'{label:<12}{int(inRow.sum()):>8}{int((converging & inRow).sum()):>11}'
                f'{int((toBlackDot & inRow).sum()):>11}{int((isBroken & inRow).sum()):>7}'
                f'{int((isMasked & inRow).sum()):>7}   {reasons}')

    lines = [f'{"targetType":<12}{"nDesign":>8}{"converging":>11}{"toBlackDot":>11}'
             f'{"broken":>7}{"masked":>7}   reasons']
    for targetType in ACCEPTED_TARGET_TYPES:
        inType = getCobraTargetMask(pfsConfig, [targetType])
        if inType.any():
            lines.append(row(targetType.name, inType))
    lines.append(row('TOTAL', np.ones(N_COBRAS, dtype=bool)))

    return lines


def finalize(pfsConfig, finalIteration, notConvergedDistanceThreshold=None,
             NOT_MOVE_MASK=None, atThetas=None, atPhis=None, convergenceFailed=False):
    """Finalize pfsConfig after converging, updating pfiCenter, fiberStatus, ra, dec"""

    def fetchFinalConvergence(pfs_visit_id, iteration):
        """Retrieve final cobra position in mm.

        Parameters
        ----------
        visitId : `int`
            Convergence identifier.
        """
        if iteration == -1:
            raise RuntimeError('Should never reach this point. fix it')

        db = opdb.OpDB()

        sql = ('SELECT cm.pfs_visit_id, cm.iteration, cm.cobra_id, cm.spot_id, cm.pfi_center_x_mm, cm.pfi_center_y_mm '
               'FROM cobra_match cm LEFT OUTER JOIN mcs_data m '
               'ON m.spot_id = cm.spot_id AND m.mcs_frame_id = cm.mcs_frame_id '
               f'WHERE cm.pfs_visit_id = {pfs_visit_id} AND cm.iteration = {iteration} '
               'ORDER BY cm.cobra_id ASC')

        return db.query_dataframe(sql)

    def _fakeFinalConvergence(pfs_visit_id, iteration, nCobras=2394):
        """Create deterministic fake convergence dataframe."""

        cobraId = np.arange(1, nCobras + 1, dtype=np.int64)

        df = pd.DataFrame({
            'pfs_visit_id': np.full(nCobras, pfs_visit_id, dtype=np.int64),
            'iteration': np.full(nCobras, iteration, dtype=np.int64),
            'cobra_id': cobraId,
            'spot_id': np.full(nCobras, -1, dtype=np.int64),
            'pfi_center_x_mm': np.full(nCobras, np.nan, dtype=np.float64),
            'pfi_center_y_mm': np.full(nCobras, np.nan, dtype=np.float64)
        })

        return df

    logger = logging.getLogger('pfsConfig')

    # Retrieve cobra_match dataset.
    lastIteration = fetchFinalConvergence(pfsConfig.visit, finalIteration)

    if lastIteration is None or not len(lastIteration):
        logger.warning(f'Could not find cobra_match data for {pfsConfig.visit} iteration={finalIteration}.')
        lastIteration = _fakeFinalConvergence(pfsConfig.visit, finalIteration)
        NO_MATCH_STATUS = FiberStatus.UNKNOWN
    else:
        NO_MATCH_STATUS = FiberStatus.BLACKSPOT

    # Setting CONVERGENCE_FAILED bit.
    if convergenceFailed:
        pfsConfig.setInstrumentStatusFlag(InstrumentStatusFlag.CONVERGENCE_FAILED)
        logger.warning('Convergence failed; Setting pfsConfig with CONVERGENCE_FAILED flag')

    # Construct the index.
    cobraId = lastIteration.cobra_id.to_numpy()
    fiberId = FiberIds().cobraIdToFiberId(cobraId)
    lastIteration['fiberId'] = fiberId
    fiberIndex = pd.DataFrame(dict(fiberId=pfsConfig.fiberId, tindex=np.arange(len(pfsConfig.fiberId))))
    fiberIndex = fiberIndex.set_index('fiberId').loc[fiberId].tindex.to_numpy()

    # Compute cobraTheta, cobraPhi in degrees and attach to the dataframe.
    atThetas = np.full(len(lastIteration), np.nan, dtype=np.float32) if atThetas is None else atThetas
    atPhis = np.full(len(lastIteration), np.nan, dtype=np.float32) if atPhis is None else atPhis
    lastIteration['cobraTheta'] = np.rad2deg(atThetas)
    lastIteration['cobraPhi'] = np.rad2deg(atPhis)

    # Setting missing matches to NaNs.
    NO_MATCH_MASK = lastIteration.spot_id.to_numpy() == -1
    colsToNan = ['pfi_center_x_mm', 'pfi_center_y_mm', 'cobraTheta', 'cobraPhi']
    lastIteration.loc[NO_MATCH_MASK, colsToNan] = np.nan

    # Set final cobra position.
    pfsConfig.pfiCenter[fiberIndex, 0] = lastIteration.pfi_center_x_mm.to_numpy()
    pfsConfig.pfiCenter[fiberIndex, 1] = lastIteration.pfi_center_y_mm.to_numpy()
    logger.info(f'{pfsConfig.filename} pfiCenter updated successfully...')

    # Set cobraTheta, cobraPhi in degrees.
    pfsConfig.cobraTheta[fiberIndex] = lastIteration.cobraTheta.to_numpy()
    pfsConfig.cobraPhi[fiberIndex] = lastIteration.cobraPhi.to_numpy()
    logger.info(f'{pfsConfig.filename} cobraTheta cobraPhi updated successfully...')

    # overriding cobraId
    pfsConfig.cobraId[fiberIndex] = cobraId
    logger.info(f'{pfsConfig.filename} cobraId updated successfully...')

    # Calculate distance to target.
    distanceToTarget = np.hypot(pfsConfig.pfiNominal[:, 0] - pfsConfig.pfiCenter[:, 0],
                                pfsConfig.pfiNominal[:, 1] - pfsConfig.pfiCenter[:, 1])

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

    # Set fiberStatus to BLACKSPOT for the not matched cobras.
    lastIteration.loc[FIBER_GOOD_MASK & NO_MATCH_MASK, 'fiberStatus'] = NO_MATCH_STATUS

    # setting MASKED fiberStatus
    if NOT_MOVE_MASK is not None:
        lastIteration.loc[FIBER_GOOD_MASK & NOT_MOVE_MASK, 'fiberStatus'] = FiberStatus.MASKED
    else:
        NOT_MOVE_MASK = np.zeros(len(lastIteration), dtype='bool')

    # setting NOTCONVERGED fiberStatus for cobra above distance threshold.
    if notConvergedDistanceThreshold is not None:
        ABOVE_THRESH_MASK = lastIteration.distanceToTarget.to_numpy() > notConvergedDistanceThreshold
        NOT_CONVERGED_MASK = ABOVE_THRESH_MASK & WITH_TARGET_MASK & FIBER_GOOD_MASK & ~NO_MATCH_MASK & ~NOT_MOVE_MASK
        lastIteration.loc[NOT_CONVERGED_MASK, 'fiberStatus'] = FiberStatus.NOTCONVERGED

    pfsConfig.fiberStatus[fiberIndex] = lastIteration.fiberStatus.to_numpy()
    logger.info(f'{pfsConfig.filename} fiberStatus updated successfully...')

    # Setting ra,dec for UNASSIGNED target.
    try:
        radec = griddata(np.vstack((lastIteration.pfi_nominal_x_mm.to_numpy()[WITH_TARGET_MASK],
                                    lastIteration.pfi_nominal_y_mm.to_numpy()[WITH_TARGET_MASK])).transpose(),
                         np.vstack((lastIteration.ra.to_numpy()[WITH_TARGET_MASK],
                                    lastIteration.dec.to_numpy()[WITH_TARGET_MASK])).transpose(),
                         np.vstack((lastIteration.pfi_center_x_mm.to_numpy()[UNASSIGNED_TARGET_MASK],
                                    lastIteration.pfi_center_y_mm.to_numpy()[UNASSIGNED_TARGET_MASK])).transpose(),
                         method='cubic')
    except ValueError:
        radec = np.vstack((lastIteration.ra.to_numpy()[UNASSIGNED_TARGET_MASK],
                           lastIteration.dec.to_numpy()[UNASSIGNED_TARGET_MASK])).transpose()

    lastIteration.loc[UNASSIGNED_TARGET_MASK, ['ra', 'dec']] = radec
    pfsConfig.ra[fiberIndex] = lastIteration.ra.to_numpy()
    pfsConfig.dec[fiberIndex] = lastIteration.dec.to_numpy()

    if any(~np.isnan(radec.ravel())):
        nRa, nDec = np.sum(~np.isnan(radec), axis=0)
        logger.info(f'{pfsConfig.filename} successfully recovered nRa({nRa}) nDec({nDec}) for UNASSIGNED target...')

    return lastIteration.iteration.max()


def writePfsConfig(pfsConfig, cmd=None):
    """Write final pfsConfig."""
    ret = pfsConfigUtils.writePfsConfig(pfsConfig)

    if cmd:
        cmd.inform(f'text="{pfsConfig.filename} written to disk"')

    return ret


def ingestPfsConfig(pfsConfig, cmd=None, **kwargs):
    """Ingest PfsConfig file in opdb tables."""
    ret = ingestPfsDesign.ingestPfsConfig(pfsConfig, **kwargs)

    if cmd:
        cmd.inform('text="pfsConfig successfully inserted in opdb !"')

    return ret
